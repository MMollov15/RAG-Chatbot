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
from langchain_community.retrievers import EnsembleRetriever 

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
    """Load PDF and split into chunks"""
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
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

def create_qa_chain(vectorstore):
    """Create the question-answering chain manually"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build chain manually using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def load_document():
    """Helper function to load and process a document"""
    while True:
        pdf_path = input("\nEnter the path to your PDF document: ").strip()
        
        if os.path.exists(pdf_path):
            chunks = load_and_process_documents(pdf_path)
            vectorstore = create_vector_store(chunks)
            qa_chain, retriever = create_qa_chain(vectorstore)
            print(f"\n✓ Document loaded: {os.path.basename(pdf_path)}")
            return qa_chain, retriever, pdf_path
        else:
            print("File not found! Please try again.")

def chat():
    """Main chat loop"""
    print("\n" + "="*50)
    print("Hi! I am your personal RAG Document chatbot")
    print("="*50)
    print("\n Feel free to load in  document and ask me any question about it." )
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
            print("\n--- Loading New -Document ---")
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
                print(f"\n[{i}] Page {doc.metadata.get('page', 'unknown')}:")
                print(doc.page_content[:200] + "...")

if __name__ == "__main__":
    chat()