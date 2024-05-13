import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    # Remove non-alphabetic characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text


def tokenize_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    print(tokens)
    return tokens


def lowercase_text(tokens):
    # Convert tokens to lowercase
    lowercase_tokens = [token.lower() for token in tokens]
    return lowercase_tokens


def remove_stopwords(tokens):
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def remove_special_characters(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def stem_text(tokens):
    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def lemmatize_text(tokens):
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def handle_emoticons_emoji(text):
    text = re.sub(r'😊|😂', '', text)
    return text


def preprocess_text(input_text):
    cleaned_text = clean_text(input_text)
    tokens = tokenize_text(cleaned_text)
    lowercased_tokens = lowercase_text(tokens)
    tokens_without_stopwords = remove_stopwords(lowercased_tokens)
    text_without_special_chars = remove_special_characters(
        tokens_without_stopwords)
    stemmed_tokens = stem_text(text_without_special_chars)
    lemmatized_tokens = lemmatize_text(stemmed_tokens)
    text_without_emoji = handle_emoticons_emoji(lemmatized_tokens)

    processed_text = " ".join(text_without_emoji)

    return processed_text


'''
def preprocess_and_combine_files(input_directory):
    # Iterate through all files in the input directory
    combined_preprocessed_text = ""
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_directory, filename)
            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                input_text = input_file.read()

            preprocessed_text = preprocess_text(input_text)
            combined_preprocessed_text += preprocessed_text + '\n'

    return combined_preprocessed_text


# Preprocess the text and combine all files into one file:
input_directory = "E:\AI & Machine Learning\Projects\CHATBOT-1\Test\cl"

output_file_path = "E:\AI & Machine Learning\Projects\CHATBOT-1\Test\out2.txt"

preprocessed_text = preprocess_and_combine_files(input_directory)

# Write the combined preprocessed text to the output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(preprocessed_text)

print(f"Preprocessed text saved to {output_file_path}")'''


def preprocess_and_save_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        input_text = input_file.read()

    preprocessed_text = preprocess_text(input_text)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(preprocessed_text)

    print(f"Preprocessed text saved to {output_file_path}")


# Preprocess the text and save to a file:
input_markdown_file = "E:\AI & Machine Learning\Projects\CHATBOT-1\Test\cl\cloud.md"
output_text_file = "E:\AI & Machine Learning\Projects\CHATBOT-1\Test\out2.txt"
