
import os
import docx
import PyPDF2
import re
import torch
import spacy
import pandas as pd
import warnings
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import numpy as np
import faiss
from typing import List
warnings.filterwarnings("ignore", category=UserWarning)


# Load pre-trained models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
nlp = spacy.load('en_core_web_md')
nlp.max_length = 1500000  # or any other desired limit

# Create an empty list to store all document text
documents = []

# Define a function to preprocess the text
def preprocess_text(text):
    if not isinstance(text, (str, bytes)):
        return None
    # Remove all non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Convert all text to lowercase
    text = text.lower()
    # Split the text into a list of words
    words = text.split()
    # check if the words list is empty
    if len(words) == 0:
        return None
    return words

# Loop through all files in the directory
for filename in os.listdir('gs://sheria1/supremecourt/'):

    # Check if the file is a PDF
    if filename.endswith('.pdf'):
        # Open the PDF file
        with open(os.path.join('gs://sheria1/supremecourt/', filename), 'rb') as f:
            #check if the file starts with !DOC
            try:
                pdfReader = PyPDF2.PdfFileReader(f)
            except PyPDF2._utils.PdfStreamError:
                warnings.warn(f"File{filename} starts with <!Doc")    
            # Get number of pages in the pdf
            num_pages = pdfReader.getNumPages()
            # Create a PyPDF2.PdfFileReader object
            pdf_reader = PyPDF2.PdfFileReader(f)
            
            # Loop through each page in the PDF and extract the text
            text = ""
            for page in range(num_pages):
                text += pdf_reader.getPage(page).extractText()
            # Preprocess the text
            words = preprocess_text(text)
            # Tokenize the input text data using the tokenizer
            if not words:
                continue
            tokens = tokenizer(" ".join(words), padding=True, truncation=True, return_tensors="pt")
            # Use the pre-trained language model to generate embeddings for the input text data
            input_tensor = torch.tensor(tokens['input_ids'])
            with torch.no_grad():
                embeddings =  model(input_tensor, tokens['attention_mask'])[0][:, 0, :]
            # Add the embeddings text to the list
            if not embeddings.detach().numpy().size:
                continue
            #convert embeddings to numpy array and append to documents list
            doc = nlp(" ".join(words))
            doc.user_data['vector'] =embeddings.detach().numpy()
            documents.append(doc)
    # Check if the file is a DOCX
    elif filename.endswith('.docx'):
        try:
            doc = docx.Document(os.path.join('gs://sheria1/supremecourt/', filename))
        except docx.exceptions.PackageNotFoundError:
            warnings.warn(f"File {filename} is not a valid DOCX file")
            continue
        # Extract text from the document
        text = ""
        for para in doc.paragraphs:
            text += para.text
        
        # Preprocess the text
        words = preprocess_text(text)
        
        # Tokenize the input text data using the tokenizer
        if not words:
            continue
        tokens = tokenizer(" ".join(words), padding=True, truncation=True, return_tensors="pt")
        
        # Use the pre-trained language model to generate embeddings for the input text data
        input_tensor = torch.tensor(tokens['input_ids'])
        with torch.no_grad():
            embeddings =  model(input_tensor, tokens['attention_mask'])[0][:, 0, :]
        
        # Add the embeddings text to the list
        if not embeddings.detach().numpy().size:
            continue
        doc = nlp(" ".join(words))
        doc.user_data['vector'] = embeddings.detach().numpy()
        documents.append(doc)
 

    # If the file is neither a PDF nor a DOCX, skip it
    else:
        continue
# Concatenate embeddings into a single numpy array
# Concatenate embeddings into a single numpy array
document_embeddings = np.concatenate([doc.user_data['vector'] for doc in documents if isinstance(doc, spacy.tokens.doc.Doc) and doc.user_data.get('vector') is not None])




# Create a list to store the preprocessed text for each document
doc_text = []
document_data = pd.DataFrame(torch.from_numpy(document_embeddings), columns=[f"feat_{i}"for i in range(document_embeddings.shape[1])])
# Loop through each row in the document_data DataFrame
for i, row in document_data.iterrows():
    # Extract the preprocessed text for the current document
    words = preprocess_text(text)
    # Append the preprocessed text to the doc_text list
    doc_text.append(" ".join(words))

# Get the labels for each document using spaCy's Named Entity Recognition (NER) model
labels = []
for doc in nlp.pipe(doc_text):
    doc_labels = [ent.label_ for ent in doc.ents]
    labels.append(doc_labels if doc_labels else [])
# convert the list of labels into pandas DataFrame
labels_df = pd.DataFrame(labels, columns=['label_1', 'label_2', 'label_3'])

# Combine the embeddings and labels into a single DataFrame

document_data = pd.concat([document_data, labels_df], axis=1)

# Train a Word2Vec model on the preprocessed documents
model =Word2Vec(sentences=doc_text, window=10, min_count=2, workers=10, sg=20)



# Load document embeddings
document_embeddings = ...

# Define similarity search engine
index = faiss.IndexFlatIP(document_embeddings.shape[1])
index.add(document_embeddings)

# Define function to preprocess user query
def preprocess_query(query: str) -> List[str]:
    words = preprocess_text(query)
    return words

# Define function to retrieve most similar documents
def semantic_search(query: str, k: int = 10,generation_model=None) -> List[str]:
    # Preprocess user query
    query_words = preprocess_query(query)
    
    # Generate query embeddings
    query_tokens = tokenizer(" ".join(query_words), padding=True, truncation=True, return_tensors="pt")
    query_input = torch.tensor(query_tokens['input_ids'])
    with torch.no_grad():
        query_embedding =  model(query_input, query_tokens['attention_mask'])[0][:, 0, :].detach().numpy()
    
    if generation_model:
        generated_text = generation_model.generate(query)
        # Add the generated text to document embeddings
        document_embeddings = np.concatenate(document_embeddings, generated_text)
        # Upadate the search index with new embeddings
        index.add(document_embeddings[-1].reshape(1, -1))
    # Pass query embeddings to similarity search engine
    distances, indices = index.search(query_embedding, k)
    
    # Return most similar documents
    results = []
    for i in indices[0]:
        results.append(doc_text[i])
    
    return results

# Define a query
query = "The city council of Pretoria vs Walker ? "

# Perform semantic search
results = semantic_search(query, k=10)

# Print the most similar documents
for doc in results:
    print(doc)
