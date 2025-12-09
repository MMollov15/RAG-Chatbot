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
import glob
#from langchain.retrievers import EnsembleRetriever

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

def load_and_process_documents(pdf_paths):
    """Load one or multiple PDFs and split into chunks with improved strategy"""
    # Handle both single file and list of files
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]
    
    all_chunks = []
    
    for pdf_path in pdf_paths:
        print(f"Loading document: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Improved text splitter - sentence-aware with better boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
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
            chunk.metadata['chunk_id'] = f"{os.path.basename(pdf_path)}_{i}"
            chunk.metadata['source_file'] = os.path.basename(pdf_path)
            if 'page' not in chunk.metadata:
                chunk.metadata['page'] = 'unknown'
        
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks")
    
    print(f"Total chunks across all documents: {len(all_chunks)}")
    
    return all_chunks

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
    """Create the question-answering chain with hybrid search and memory"""
    # Create custom hybrid retriever
    hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        chunks=chunks,
        semantic_weight=0.5,
        keyword_weight=0.5
    )
    
    # Template that includes conversation history
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Conversation History:
{chat_history}

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
        {"context": RunnablePassthrough() | retrieve_and_format, 
         "question": RunnablePassthrough(),
         "chat_history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, hybrid_retriever

def load_document():
    """Helper function to load and process document(s)"""
    print("\nOptions:")
    print("  1. Load a single PDF file")
    print("  2. Load all PDFs from a folder")
    print("  3. Load multiple specific PDF files")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    pdf_paths = []
    
    if choice == '1':
        # Single file
        while True:
            pdf_path = input("\nEnter the path to your PDF document: ").strip()
            if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
                pdf_paths = [pdf_path]
                break
            else:
                print("File not found or not a PDF! Please try again.")
    
    elif choice == '2':
        # Folder
        while True:
            folder_path = input("\nEnter the folder path: ").strip()
            if os.path.isdir(folder_path):
                pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
                if pdf_paths:
                    print(f"\nFound {len(pdf_paths)} PDF files:")
                    for path in pdf_paths:
                        print(f"  - {os.path.basename(path)}")
                    confirm = input("\nLoad all these files? (y/n): ").strip().lower()
                    if confirm == 'y':
                        break
                    else:
                        pdf_paths = []
                else:
                    print("No PDF files found in this folder!")
            else:
                print("Folder not found! Please try again.")
    
    elif choice == '3':
        # Multiple specific files
        print("\nEnter PDF file paths one by one. Type 'done' when finished.")
        while True:
            pdf_path = input("PDF path (or 'done'): ").strip()
            if pdf_path.lower() == 'done':
                if pdf_paths:
                    break
                else:
                    print("No files added yet! Add at least one file.")
            elif os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
                pdf_paths.append(pdf_path)
                print(f"  ✓ Added: {os.path.basename(pdf_path)}")
            else:
                print("  ✗ File not found or not a PDF!")
    
    else:
        print("Invalid choice! Defaulting to single file mode.")
        return load_document()
    
    # Process the document(s)
    chunks = load_and_process_documents(pdf_paths)
    vectorstore = create_vector_store(chunks)
    qa_chain, retriever = create_qa_chain(vectorstore, chunks)
    
    if len(pdf_paths) == 1:
        print(f"\n✓ Document loaded with hybrid search: {os.path.basename(pdf_paths[0])}")
        doc_name = os.path.basename(pdf_paths[0])
    else:
        print(f"\n✓ {len(pdf_paths)} documents loaded with hybrid search")
        doc_name = f"{len(pdf_paths)} documents"
    
    return qa_chain, retriever, doc_name

def format_chat_history(history, max_exchanges=5):
    """Format chat history for the prompt, keeping only recent exchanges"""
    if not history:
        return "No previous conversation."
    
    # Keep only the last N exchanges
    recent_history = history[-(max_exchanges * 2):]
    
    formatted = []
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)

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