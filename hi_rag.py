'''
This is an example of building a hierarchical RAG.

Referece: https://github.com/NirDiamant/RAG_Techniques/tree/main/all_rag_techniques
'''

import os
import sys
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from PIL import Image
import pdb

def encode_pdf_hierarchical(path, chunk_size=200, chunk_overlap=0, is_string=False):
    """
    Encodes a PDF into a hierarchical vector store using embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A tuple containing two Chroma vector stores:
        1. Document-level summaries
        2. Detailed chunks
    """

    # Load PDF documents
    if not is_string:
        loader = PyPDFLoader(path)
        documents = loader.load()
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents([path])

    # Create document-level summaries
    summary_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=4000, timeout=None)
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    def summarize_doc(doc):
        """
        Summarizes a single document with rate limit handling.

        Args:
            doc: The document to be summarized.
            
        Returns:
            A summarized Document object.
        """
        #summary_output = summary_chain.invoke([doc])
        #summary = summary_output['output_text']
        summary = doc.page_content
        return Document(
            page_content=summary,
            metadata={"source": path, "page": doc.metadata["page"], "summary": True}
        )

    # Process documents in smaller batches
    batch_size = 1
    summaries = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print("Summarizing page {}......".format(i))
        batch_summaries = [summarize_doc(doc) for doc in batch]
        summaries.extend(batch_summaries)

    # Split documents into detailed chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    detailed_chunks = text_splitter.split_documents(documents)

    # Update metadata for detailed chunks
    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "summary": False,
            "page": int(chunk.metadata.get("page", 0))
        })

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create vector stores asynchronously with rate limit handling
    def create_vectorstore(docs, chroma_path):
        """
        Creates a vector store from a list of documents.
        
        Args:
            docs: The list of documents to be embedded.
            chroma_path: The path to the Chroma vector store.
        Returns:
            A Chroma vector store containing the embedded documents.
        """
        return Chroma.from_documents(docs, embeddings, persist_directory=chroma_path, collection_metadata={"hnsw:space": "cosine"})

    # Generate vector stores for summaries and detailed chunks concurrently
    summary_vectorstore, detailed_vectorstore = create_vectorstore(summaries, "./vector_stores/summary_store"), create_vectorstore(detailed_chunks, "./vector_stores/detailed_store")

    return summary_vectorstore, detailed_vectorstore

def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=3):
    """
    Performs a hierarchical retrieval using the query.

    Args:
        query: The search query.
        summary_vectorstore: The vector store containing document summaries.
        detailed_vectorstore: The vector store containing detailed chunks.
        k_summaries: The number of top summaries to retrieve.
        k_chunks: The number of detailed chunks to retrieve per summary.

    Returns:
        A list of relevant detailed chunks.
    """

    # Retrieve top summaries
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)
    relevant_chunks = []
    for summary in top_summaries:
        # For each summary, retrieve relevant detailed chunks
        page_number = summary.metadata["page"]
        page_filter = {"page": page_number}
        page_chunks = detailed_vectorstore.similarity_search(
            query, 
            k=k_chunks, 
            filter=page_filter
        )
        relevant_chunks.extend(page_chunks)

    return relevant_chunks

# Load environment variables from a .env file
load_dotenv()

# define pdf path
doc_path = "./pdf/ssvod.pdf"

# encode the PDF to both document-level summaries and detailed chunks if the vector stores do not exist
summary_store, detailed_store = encode_pdf_hierarchical(doc_path)

# testing
query = "Tell me about Tanvir, Burhan, and Chun-Hao Liu."
results = retrieve_hierarchical(query, summary_store, detailed_store)

# Print results
for chunk in results:
    print(f"Page: {chunk.metadata['page']}")
    print(f"Content: {chunk.page_content}...")
    print("---")