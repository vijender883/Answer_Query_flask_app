import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Define directories
DOCS_DIR = "./documents"
LINKEDIN_RESUME_DIR = "./linkedin_resumes"

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(LINKEDIN_RESUME_DIR, exist_ok=True)

# Set Google API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=pinecone_api_key)

# Initialize Embeddings
print("Initializing embeddings model...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)
print("Embeddings model initialized.")

cloud = os.getenv("PINECONE_CLOUD") or "aws"
region = os.getenv("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)

# Check if index name is provided
index_name = os.getenv("PINECONE_INDEX")
if not index_name:
    raise ValueError("PINECONE_INDEX environment variable not found or is empty")

# Check if index exists and create if needed
print(f"\nChecking for index '{index_name}'...")
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating index...")
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension for Gemini embedding model
        metric="cosine",
        spec=spec
    )
    print(f"Index '{index_name}' created.")
    # Give index a moment to become ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print(f"Index '{index_name}' is ready.")
else:
    print(f"Index '{index_name}' found.")

index = pc.Index(index_name)

# Delete all existing vectors in the index
try:
    index_stats_before_delete = index.describe_index_stats()
    print("\nIndex stats before potential deletion:")
    print(index_stats_before_delete)

    if index_stats_before_delete.total_vector_count > 0:
        print(f"\nIndex '{index_name}' contains {index_stats_before_delete.total_vector_count} vectors. Deleting all...")
        index.delete(delete_all=True)
        print("All vectors deleted.")
    else:
        print(f"\nIndex '{index_name}' is already empty. Skipping deletion.")

    # Give index a moment to register deletion
    time.sleep(10)

    index_stats_before_upsert = index.describe_index_stats()
    print("Index stats before upsert:")
    print(index_stats_before_upsert)

except Exception as e:
    print(f"\nError accessing index stats or deleting vectors: {e}")
    print("Proceeding with upsert, but index state might be inconsistent.")

# Process Event Documents from documents folder
print(f"\n--- Processing Event Documents from '{DOCS_DIR}' ---")
if not os.path.exists(DOCS_DIR):
    print(f"Error: Documents directory '{DOCS_DIR}' not found. Skipping event documents.")
    event_documents = []
else:
    print(f"Loading documents from '{DOCS_DIR}'...")
    try:
        loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        event_documents = loader.load()
        print(f"Loaded {len(event_documents)} event documents.")
        if not event_documents:
            print("No PDF documents found in the event documents directory.")
    except Exception as e:
        print(f"Error loading event documents: {e}")
        event_documents = []

event_chunks = []
if event_documents:
    # Split event documents into chunks
    print(f"Splitting {len(event_documents)} event documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
    event_chunks = text_splitter.split_documents(event_documents)
    print(f"Created {len(event_chunks)} event chunks.")

    if not event_chunks:
        print("No event chunks created after splitting.")

# Upsert event document chunks
if event_chunks:
    print(f"Upserting {len(event_chunks)} event document chunks to Pinecone index '{index_name}'...")
    try:
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            text_key="text"
        )
        vectorstore.add_documents(event_chunks)
        print(f"Successfully uploaded {len(event_chunks)} event document chunks.")
    except Exception as e:
        print(f"Error during upserting event documents to Pinecone: {e}")

# Process LinkedIn Resume Documents
print(f"\n--- Processing LinkedIn Resume Documents from '{LINKEDIN_RESUME_DIR}' ---")

resume_files = [f for f in os.listdir(LINKEDIN_RESUME_DIR) if f.endswith('.pdf')]
if not resume_files:
    print(f"No PDF files found in the '{LINKEDIN_RESUME_DIR}' directory. Skipping resume processing.")
else:
    print(f"Found {len(resume_files)} LinkedIn resume files.")
    processed_count = 0
    error_count = 0

    for resume_file in resume_files:
        resume_file_path = os.path.join(LINKEDIN_RESUME_DIR, resume_file)
        print(f"\nProcessing LinkedIn resume: {resume_file}")
        
        # Extract userId from filename (remove .pdf extension)
        user_id = os.path.splitext(resume_file)[0]
        
        try:
            # Load the resume document
            loader = PyPDFLoader(resume_file_path)
            resume_documents = loader.load()
            
            if not resume_documents:
                print(f"  Warning: No content extracted from '{resume_file}'. Skipping.")
                continue
                
            # Add userId metadata to each document
            for doc in resume_documents:
                doc.metadata["userId"] = user_id
                
            # Split the resume document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
            resume_chunks = text_splitter.split_documents(resume_documents)
            print(f"  Split into {len(resume_chunks)} chunks with userId: {user_id}")
            
            if not resume_chunks:
                print(f"  Warning: No chunks created after splitting '{resume_file}'. Skipping.")
                continue
                
            # Upsert the resume chunks to Pinecone
            print(f"  Upserting {len(resume_chunks)} chunks for '{resume_file}' with userId metadata...")
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                text_key="text"
            )
            vectorstore.add_documents(resume_chunks)
            
            print(f"  Successfully uploaded chunks for '{resume_file}'.")
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing and embedding '{resume_file}': {e}")
            error_count += 1
            
    print(f"\nLinkedIn resume processing summary:")
    print(f"  Successfully processed and added {processed_count} resumes to Pinecone.")
    print(f"  Encountered errors during processing/embedding for {error_count} resumes.")

# Final Check
print("\nFinal Index stats after all upserts:")
try:
    print(index.describe_index_stats())
except Exception as e:
    print(f"Error getting final index stats: {e}")

print("\nIngestion process finished.")