import os
from dotenv import load_dotenv
from gym_guide_chatbot.utils import load_pdf_file, download_embedding_model, text_split, clean_text
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()


#Get all the necessary data
data_dir = os.getenv("DATA_PATH")
print("Extracting Data...\n")
extracted_data = load_pdf_file(data=data_dir)
print("Creating Chunks...\n")
chunks = text_split(extracted_data=extracted_data)
print("Downloading Embedding model...\n")
embeddings = download_embedding_model()


# Only use this if the data is not structured while creating chunks
print("Cleaning up data...\n")
for doc in chunks:
    doc.page_content = clean_text(doc.page_content)


#Initialize pinecone and create index if not already

print("Creating PineCone Index...\n")
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = os.getenv("INDEX_NAME")


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
else:
    print(f"Pinecone index '{index_name}' already exists. Skipping creation.")

# Create embeddings
print("Creating Vector Embeddings...\n")
vector_embeddings = PineconeVectorStore.from_documents(
    index_name=index_name,
    documents=chunks,
    embedding=embeddings
)

