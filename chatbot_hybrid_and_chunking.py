import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

# Load environment variables
load_dotenv()

# Initialize the embedding model (runs locally)
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize the LLM (Groq API)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def load_and_process_documents(pdf_path):
    """Load PDF and split into chunks with improved strategy"""
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Improved text splitter - sentence-aware with better boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Slightly smaller for better precision
        chunk_overlap=150,  # More overlap to preserve context
        length_function=len,
        separators=[
            "\n\n",  # Double newline (paragraphs) - highest priority
            "\n",    # Single newline
            ". ",    # Sentence endings
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Characters (last resort)
        ],
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Enhance chunks with additional metadata
    for i, chunk in enumerate(chunks):
        # Add chunk index for reference
        chunk.metadata['chunk_id'] = i
        # Add document name for multi-doc support later
        chunk.metadata['source_file'] = os.path.basename(pdf_path)
        # Preserve page number from PDF loader
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 'unknown'
    
    print(f"Split into {len(chunks)} chunks with enhanced metadata")
    
    return chunks

def create_vector_store(chunks):
    """Create and populate vector database"""
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created!")
    return vectorstore

def format_docs(docs):
    """Format retrieved documents into a string"""
    return "\n\n".join(doc.page_content for doc in docs)

class HybridRetriever:
    """Custom hybrid retriever combining semantic and keyword search"""
    
    def __init__(self, vectorstore, chunks, semantic_weight=0.5, keyword_weight=0.5):
        self.semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        self.keyword_retriever = BM25Retriever.from_documents(chunks)
        self.keyword_retriever.k = 5
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def invoke(self, query):
        """Retrieve documents using hybrid search"""
        # Get results from both retrievers
        semantic_docs = self.semantic_retriever.invoke(query)
        keyword_docs = self.keyword_retriever.invoke(query)
        
        # Combine and deduplicate documents
        # Use a dict to track unique documents by their content
        doc_scores = {}
        
        # Score semantic results (higher position = higher score)
        for i, doc in enumerate(semantic_docs):
            score = (len(semantic_docs) - i) / len(semantic_docs) * self.semantic_weight
            key = doc.page_content[:100]  # Use first 100 chars as key
            if key in doc_scores:
                doc_scores[key]['score'] += score
            else:
                doc_scores[key] = {'doc': doc, 'score': score}
        
        # Score keyword results (higher position = higher score)
        for i, doc in enumerate(keyword_docs):
            score = (len(keyword_docs) - i) / len(keyword_docs) * self.keyword_weight
            key = doc.page_content[:100]
            if key in doc_scores:
                doc_scores[key]['score'] += score
            else:
                doc_scores[key] = {'doc': doc, 'score': score}
        
        # Sort by combined score and return top documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs[:5]]

def create_qa_chain(vectorstore, chunks):
    """Create the question-answering chain with hybrid search"""
    # Create custom hybrid retriever
    hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        chunks=chunks,
        semantic_weight=0.5,
        keyword_weight=0.5
    )
    
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create a function that wraps the retriever
    def retrieve_and_format(question):
        docs = hybrid_retriever.invoke(question)
        return format_docs(docs)
    
    # Build chain manually using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": RunnablePassthrough() | retrieve_and_format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, hybrid_retriever

def load_document():
    """Helper function to load and process a document"""
    while True:
        pdf_path = input("\nEnter the path to your PDF document: ").strip()
        
        if os.path.exists(pdf_path):
            chunks = load_and_process_documents(pdf_path)
            vectorstore = create_vector_store(chunks)
            qa_chain, retriever = create_qa_chain(vectorstore, chunks)
            print(f"\n✓ Document loaded with hybrid search: {os.path.basename(pdf_path)}")
            return qa_chain, retriever, pdf_path
        else:
            print("File not found! Please try again.")

def chat():
    """Main chat loop"""
    print("\n" + "="*50)
    print("RAG Document Chatbot")
    print("="*50)
    print("\nCommands:")
    print("  /load  - Load a new document")
    print("  quit   - Exit the chatbot")
    
    # Load initial document
    qa_chain, retriever, current_doc = load_document()
    
    print("\n✓ Ready to chat!\n")
    
    # Chat loop
    while True:
        question = input("You: ").strip()
        
        # Check for quit command
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Check for load command
        if question.lower() == '/load':
            print("\n--- Loading New Document ---")
            qa_chain, retriever, current_doc = load_document()
            print("--- Ready to answer questions about the new document ---\n")
            continue
        
        # Skip empty input
        if not question:
            continue
        
        # Answer the question
        print("\nThinking...")
        answer = qa_chain.invoke(question)
        
        print(f"\nBot: {answer}\n")
        print(f"[Current document: {os.path.basename(current_doc)}]")
        
        # Optionally show sources
        show_sources = input("Show sources? (y/n): ").strip().lower()
        if show_sources == 'y':
            docs = retriever.invoke(question)
            print("\nSources:")
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get('page', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                source_file = doc.metadata.get('source_file', 'unknown')
                print(f"\n[{i}] File: {source_file} | Page: {page} | Chunk: {chunk_id}")
                print(f"Content preview: {doc.page_content[:250]}...")
                print("-" * 50)

if __name__ == "__main__":
    chat()