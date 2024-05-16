# from flask import Flask, render_template, jsonify, request
# from src.helper import encode_text_chunks
# from pinecone import Pinecone, ServerlessSpec
# import pinecone
# from dotenv import load_dotenv
# from langchain_community.llms import ctransformers
# from langchain.chains import RetrievalQA
# from src.prompt import *
# from langchain import PromptTemplate
# from sentence_transformers import SentenceTransformer
# from langchain_pinecone import PineconeVectorStore
# from langchain.retrievers import SelfQueryRetriever
# import os

# app = Flask(__name__)

# # Load environment variables from the .env file
# load_dotenv()

# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index("mymchatbot")

# namespace = "book1"





# # PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# # chain_type_kwargs={"prompt": PROMPT}

# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# llm = ctransformers.CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                                   model_type="llama",
#                                   config={'max_new_tokens': 512,
#                                           'temperature': 0.8})



# #******************Initialize the RetrievalQA chain****************


# # Assuming this is your sentence transformer model
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Create the PineconeVectorStore instance (assuming you have the index details)
# vectorstore = PineconeVectorStore(index, model)



# document_content_description = "page_content"
# metadata_field_info = {}  # Define the metadata_field_info variable (if needed)

# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectorstore,
#     document_content_description,
#     metadata_field_info,
#     enable_limit=True,
#     verbose=True,
# )

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",  # Replace "stuff" with your actual chain type name
#     retriever=retriever,
# )


# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     result=qa({"query": input})
#     print("Response : ", result["result"])
#     return str(result["result"])


# if __name__ == '__main__':
#     app.run(debug= True)




















from flask import Flask, render_template, jsonify, request
from src.helper import encode_text_chunks
from pinecone import Pinecone, ServerlessSpec
import pinecone
from dotenv import load_dotenv
from langchain_community.llms import ctransformers
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import SelfQueryRetriever
import os

app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("mymchatbot")

namespace = "book1"





PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = ctransformers.CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                                  model_type="llama",
                                  config={'max_new_tokens': 512,
                                          'temperature': 0.8})



#******************Initialize the RetrievalQA chain****************


# Assuming this is your sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create the PineconeVectorStore instance (assuming you have the index details)
vectorstore = PineconeVectorStore(index, model)



document_content_description = "page_content"
metadata_field_info = {}  # Define the metadata_field_info variable (if needed)

MAX_CONTEXT_LENGTH = 256  # Adjust this value as needed

def get_context(conversation_history):
  # Implement logic to select the most recent messages for context
  # Ensure the total token count of selected messages is less than MAX_CONTEXT_LENGTH
  context = "..."  # Build the context string based on selected messages
  return context

conversation_history = []  # Define the conversation_history variable

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
    context=get_context(conversation_history=4)  # Pass the limited context
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Replace "stuff" with your actual chain type name
    retriever=retriever,
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug= True)

