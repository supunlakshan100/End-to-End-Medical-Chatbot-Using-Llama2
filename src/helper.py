from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Libraries for preprocess the text data
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')





#***************Load PDF files from a directory***************

def load_pdf(data):
    loader=DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

#***************preprocess_documents*************

def preprocess_documents(extracted_data):
    """
    Preprocesses the documents in extracted_data.

    Args:
    - extracted_data: List of documents to preprocess.

    Returns:
    - preprocessed_documents: List of preprocessed documents.
    """

    # Define preprocessing functions
    def clean_text(text):
        # Remove non-alphanumeric characters but keep URLs intact
        text = re.sub(r'[^\w\s\./-]', '', text)
        text = ' '.join(text.split())
        return text

    def lowercase_text(text):
        # Convert text to lowercase
        return text.lower()

    def remove_stopwords(text):
        # Remove stopwords from the text (excluding URLs)
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words and not re.match(r'^https?://', token)]
        return ' '.join(filtered_tokens)

    # Apply preprocessing to each document
    preprocessed_documents = []
    for document in extracted_data:
        content = document.page_content
        # Clean text (keeping URLs)
        cleaned_content = clean_text(content)
        # Lowercase text
        lowercase_content = lowercase_text(cleaned_content)
        # Remove stopwords (excluding URLs)
        filtered_content = remove_stopwords(lowercase_content)
        # Update the 'page_content' attribute of the document with preprocessed text
        document.page_content = filtered_content
        # Append the preprocessed document to the list
        preprocessed_documents.append(document)

    return preprocessed_documents



#***************Split the preprocessed text into smaller chunks***************

def text_split(preprocessed_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(preprocessed_documents)  # Wrap extracted_data in a list
    return text_chunks




#***************Extract embeddings from the text chunks************

def encode_text_chunks(text_chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Encodes text chunks using the specified SentenceTransformer model.

    Args:
    - text_chunks: List of text chunks to encode.
    - model_name: Name or path of the SentenceTransformer model to use (default is 'sentence-transformers/all-MiniLM-L6-v2').

    Returns:
    - embeddings: Encoded embeddings for the text chunks.
    """
    # Extract text from Document objects
    text_list = [t.page_content for t in text_chunks]

    # Initialize SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Encode the text chunks
    embeddings = model.encode(text_list)

    return embeddings
    # print(embeddings)

