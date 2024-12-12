import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import openai

from dotenv import load_dotenv
load_dotenv()

# load api key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
# print(os.environ["NVIDIA_API_KEY"])


def upload_data():
    # load
    loader = PyPDFDirectoryLoader("research_papers")
    documents = loader.load()

    # split using recursive --functionality of recursive is to split the documents semantically
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, chunk_overlap = 50, separators=None
    )

    splitted_documents = text_splitter.split_documents(documents=documents)
    return splitted_documents


def create_vector_store(splitted_documents):
    try:
        embeddings = NVIDIAEmbeddings()
        # Ensure splitted_documents is not empty
        if not splitted_documents:
            raise ValueError("Your local database is empty. Please provide context.")
        
        vector_DB = FAISS.from_documents(splitted_documents, embeddings)
        vector_DB.save_local("faiss_index")
        print("Vector store created and saved successfully.")
    
    except ValueError as ve:
        print(f"Error: {ve}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def create_llm_model():
    """Creates a GPT-3.5-turbo model interface."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai

def get_response(llm, vector_DB, question, chat_history):
    """Get response from GPT-3.5-turbo using a conversational retrieval chain."""
    # Retrieve relevant documents from FAISS vector database
    retriever = vector_DB.as_retriever()
    docs = retriever.get_relevant_documents(question)
    
    # Combine retrieved documents into context
    retrieved_texts = "\n\n".join([doc.page_content for doc in docs])
    context = f"Context: {retrieved_texts}\n\n"
    
    # Format chat history for GPT-3.5-turbo
    history = "\n".join(
        [f"{role}: {message}" for role, message in chat_history]
    )
    
    # Construct the final prompt
    prompt = (
        f"{context}"
        f"Chat History:\n{history}\n\n"
        f"Question: {question}\n\n"
        f"Answer with relevant details based on the context above:"
    )
    
    response = llm.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. "},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    # Extract and return the response
    answer = response['choices'][0]['message']['content'].strip()

    print("Calling gpt_3.5 .....")
    return {"answer": answer, "source_documents": docs}



















