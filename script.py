import os
import docx
import PyPDF2
import re
import warnings
from gensim.models import Word2Vec
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)


# Create an empty list to store all document text
documents = []

# Define a function to preprocess the text
def preprocess_text(text):
    while not isinstance(text, (str, bytes)):
        continue
    # Remove all non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Convert all text to lowercase
    text = text.lower()
    # Split the text into a list of words
    words = text.split()
    return words

# Loop through all files in the directory
for filename in os.listdir('/home/piousmionki/new/supremecourt'):

    # Check if the file is a PDF
    if filename.endswith('.pdf'):
        # Open the PDF file
        with open(os.path.join('/home/piousmionki/new/supremecourt', filename), 'rb') as f:
            #check if the file starts with !DOC
            try:
                pdfReader = PyPDF2.PdfFileReader(f)
            except PyPDF2._utils.PdfStreamError:
                warnings.warn(f"File{filename} starts with <!Doc")    
            # Create a PyPDF2.PdfFileReader object
            pdf_reader = PyPDF2.PdfFileReader(f)
            
            # Loop through each page in the PDF and extract the text
            text = ""
            for page in range(pdf_reader.getNumPages()):
                text += pdf_reader.getPage(page).extractText()
            # Preprocess the text
            words = preprocess_text(text)
            # Add the document text to the list
            documents.append(words)

    # Check if the file is a DOCX
    elif filename.endswith('.docx'):
        # Open the DOCX file
        doc = docx.Document(os.path.join('', filename))
        # Loop through each paragraph in the DOCX and extract the text
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text
        # Preprocess the text
        words = preprocess_text(text)
        # Add the document text to the list
        documents.append(words)

    # If the file is neither a PDF nor a DOCX, skip it
    else:
        continue

# Train a Word2Vec model on the preprocessed documents
model = Word2Vec(documents, vector_size=150, window=10, min_count=2, workers=10, sg=1)

# Define a function to encode a document as a numerical vector
def encode_document(document):
    # Tokenize the text into individual words
    words = preprocess_text(document)
    # Look up the corresponding word vectors in the Word2Vec model
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    # Compute the average vector of the word vectors
    if len(word_vectors) > 0:
        document_vector = np.mean(word_vectors, axis=0)
    else:
        document_vector = np.zeros(model.vector_size)
    return document_vector
    # Encode each document as a numerical vector
document_vectors = [encode_document(document) for document in documents]
