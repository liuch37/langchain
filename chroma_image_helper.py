'''
Helper function for Chroma that supports customized image embedding
'''

from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np
from langchain_chroma import Chroma

class ImageEmbeddings:
    """
    A custom embedding class for LangChain that wraps a pre-trained image embedding model.
    It provides methods to embed documents (images) and queries (text or other images).
    """

    def __init__(self, model_name: str = 'clip-ViT-B-32'):
        """
        Initialize the ImageEmbeddings class with a pre-trained model.

        Args:
            model_name (str): The name of the model to use for generating embeddings.
                              Defaults to 'clip-ViT-B-32'.
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, image_paths: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for a list of image file paths.

        Args:
            image_paths (list[str]): List of paths to image files.

        Returns:
            list[list]: List of embeddings for the provided images.
        """
        embeddings = []
        for path in image_paths:
            image = Image.open(path)  # Open the image
            embedding = self.model.encode(image)  # Generate the embedding
            embeddings.append(embedding.tolist())  # Convert the embedding to a list and append to the list of embeddings
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a query, which could be text or an image.

        Args:
            query (str): A query, typically text, describing the desired content.

        Returns:
            list: Embedding for the query.
        """
        return self.model.encode(query).tolist()  # Encode the query as an embedding

def Chroma_from_images(image_paths, metadata_list, chroma_path, model_name='clip-ViT-B-32', collection_name="image_collection"):
    """
    Creates a ChromaDB vector store for a list of images and their metadata.

    Args:
        image_paths (list[str]): List of image file paths.
        metadata_list (list[dict]): List of metadata dictionaries corresponding to each image.
        chroma_path (str): Path to the ChromaDB directory.
        model_name (str): Name of the pre-trained embedding model. Defaults to 'clip-ViT-B-32'.
        collection_name (str): Name of the ChromaDB collection. Defaults to "image_collection".

    Returns:
        Chroma: A ChromaDB vector store containing the image embeddings and metadata.
    """
    # Initialize the embeddings wrapper
    embedding_function = ImageEmbeddings(model_name=model_name)

    # Generate embeddings for the images
    image_embeddings = embedding_function.embed_documents(image_paths)

    # Initialize the ChromaDB store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=chroma_path,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Add images and metadata to the store
    vector_store.add_texts(texts=image_paths, metadatas=metadata_list, embeddings=image_embeddings)

    return vector_store

if __name__ == "__main__":
    image_path = './images'
    image_paths = []
    metadata_list = []
    chroma_path = './chroma_db'
    for image_name in os.listdir(image_path):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_paths.append(os.path.join(image_path, image_name))
            metadata_list.append({"filename": image_name, "description": "Dummy"})

    # Create a Chroma vector store from images
    vector_store = Chroma_from_images(image_paths, metadata_list, chroma_path)

    # Query the store
    query = "a cat and a dog"
    results = vector_store.similarity_search(query, k=2)

    for result in results:
        print(result.metadata)
