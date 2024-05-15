from src.helper import load_pdf,preprocess_documents,text_split,encode_text_chunks
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec


# Load environment variables from the .env file
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# print("PINECONE_API_KEY:", PINECONE_API_KEY)



extracted_data = load_pdf("/home/supunlakshan/AI PROJECTS/End-to-End-Medical-Chatbot-Using-Llama2/data")
preprocessed_documents = preprocess_documents(extracted_data)
text_chunks=text_split(preprocessed_documents)
embeddings=encode_text_chunks(text_chunks)


# #***************store the data in pinecone vector store***************


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("mymchatbot")

# Specify your namespace
namespace = "book1"

# print(index.describe_index_stats())

# Create a Pinecone ids
import uuid
ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]

# print(len(ids))
# print(ids)

# Create a list of dictionaries to upsert into the Pinecone index
vectors_to_upsert = [
    {
        "id": str(ids[i]),  # Ensure each ID is a string
        "values": embeddings[i],  # The embedding vector for the text chunk
        "metadata": {"page_content": str(text_chunks[i])}  # Storing page content as metadata
    } for i in range(len(embeddings))
]

# print(vectors_to_upsert)

# Proceed with the upsert
def upsert_in_batches(index, vectors, namespace, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=namespace)

# Upsert the vectors in batches
upsert_in_batches(index, vectors_to_upsert, namespace, batch_size=100)