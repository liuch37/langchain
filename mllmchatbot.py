'''
Build a multi-modal chatbot.
'''

import streamlit as st
import faiss
import numpy as np
from PIL import Image
from langchain_community.vectorstores import FAISS
from transformers import CLIPProcessor, CLIPModel
import torch

import fitz  # PyMuPDF for PDF handling
from PIL import Image
import io
import matplotlib.pyplot as plt
import pdb

# Extract text and images from a PDF
def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = ""
    image_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_data += page.get_text("text")

        # Extract images
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_data.append(image)

    return text_data, image_data

# Function to compute image embeddings
def compute_image_embeddings(images, model, processor):
    embeddings = []
    for idx, image in enumerate(images):
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize the embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())
        print("Processing image embeddings {}......".format(idx))

    return embeddings

# Function to compute text embeddings using CLIP
def get_text_embedding(text, model, processor):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize the embeddings
    return text_features.cpu().numpy()

# Load CLIP model for text-to-image search
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Save image embeddings to local
#image_embeddings = np.load("image_embeddings.npy")  # Precomputed embeddings
#images = np.load("extracted_images.npy")    # Associated filenames

pdf_file = './pdf/ssvod.pdf'
# extract pdf texts and images    
print("Extracting pdfs......")
text_data, images = extract_text_and_images_from_pdf(pdf_file)
print("Total {} images being extracted".format(len(images)))

# store image embeddings
print("Computing image embeddings...... ")
image_embeddings = compute_image_embeddings(images, model, processor)
image_embeddings = np.asarray(image_embeddings).squeeze()

# Create FAISS index and add image embeddings
print("Creating image embedding database......")
index = faiss.IndexFlatL2(image_embeddings.shape[1])  # L2 distance index
index.add(np.array(image_embeddings))

# Streamlit app interface
st.title("Text-to-Image Search App with Multi-Modal from a PDF")

# Get user input
user_input = st.text_input("Enter your query:", key="input")

if user_input:
    # Compute text embedding for the query
    text_embedding = get_text_embedding(user_input, model, processor)

    # Search the FAISS index for relevant images
    D, I = index.search(text_embedding, k=3)  # Retrieve top-3 images

    st.write("Top-3 images based on your query:")

    # Display the top-3 images
    for idx in I[0]:
        st.image(images[idx])

# debug
if __name__ == "__main__":
    pdf_file = './pdf/ssvod.pdf'
    # extract pdf texts and images    
    print("Extracting pdfs......")
    text_data, images = extract_text_and_images_from_pdf(pdf_file)
    print("Total {} images being extracted".format(len(images)))
    # Load CLIP model for text-to-image search
    print("Load model......")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # store image embeddings
    print("Computing image embeddings...... ")
    image_embeddings = compute_image_embeddings(images, model, processor)
    image_embeddings = np.asarray(image_embeddings).squeeze()

    # extracted images save as files
    I = []
    for image in images:
        I.append(np.array(image.resize((512,512))))        
    I = np.array(I)

    # save image embeddings
    print("Save image embeddings......")
    np.save('image_embeddings.npy', image_embeddings)
    np.save('extracted_images.npy', I)