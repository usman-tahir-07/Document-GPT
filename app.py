from langchain_community.document_loaders import CSVLoader,PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os

def chunking(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 50
    )
    splitted_text = text_splitter.split_documents(data)
    return splitted_text

def get_llm_response(query, content):
    llm_groq = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.3
    )
    template = '''
    You have to answer the user query based on the provided context.
    If user ask something that is out of context of the content,
    then guide the user of what he can ask and what you can tell him.
    User query: {query}
    Context: {content}
    Answer: 
    '''
    prompt = PromptTemplate.from_template(template=template)
    chain = prompt | llm_groq | StrOutputParser()
    response = chain.invoke({"query":query,"content":content})
    return response

def store_in_vector(chunks):
    if not chunks:
        st.error("No content to process. Please check your CSV files.")
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def retrieving(vector_store, query):
    if vector_store is None:
        return []
    similar_chunk = vector_store.similarity_search(query, k=3)
    return similar_chunk

def process_file(csv_files):
    with st.spinner("Processing..."):
        all_documents = []
        for csv_file in csv_files:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, csv_file.name)
            with open(path, "wb") as f:
                f.write(csv_file.getvalue())
            extention = Path(path).suffix
            if extention == ".csv":
                loader = CSVLoader(path)
            elif extention == ".pdf":
                loader = PyPDFLoader(path)
            try:
                data = loader.load()
                if not data:
                    st.warning(f"No data found in file: {csv_file.name}")
                all_documents.extend(data)
            except Exception as e:
                st.error(f"Error processing file {csv_file.name}: {str(e)}")
        
        if not all_documents:
            st.error("No valid data found in any of the uploaded files.")
            return None
        
        chunks = chunking(all_documents)
        vector_store = store_in_vector(chunks)
    return vector_store

def main():
    load_dotenv()
    st.set_page_config("Document Talks")
    st.header("Chit-Chat with your File")

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.subheader("Upload your File(s)")
        csv_files = st.file_uploader("Upload here...", accept_multiple_files=True)
        if csv_files and st.session_state.vector_store is None:
            st.session_state.vector_store = process_file(csv_files)
            if st.session_state.vector_store:
                st.success("File(s) Uploaded and Processed Successfully")
            else:
                st.error("Failed to process files. Please check the errors above.")

    input_text = st.text_input("Ask anything about your file(s): ")
    if input_text and st.session_state.vector_store:
        with st.spinner("Thinking..."):
            similar_chunk = retrieving(st.session_state.vector_store, input_text)
            if similar_chunk:
                response = get_llm_response(input_text, similar_chunk)
                st.write(response)
            else:
                st.warning("No relevant information found. Try a different question or upload more files.")

if __name__ == "__main__":
    main()