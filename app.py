import streamlit as st
import os 
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbedding
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader



from dotenv import load_dotenv
load_dotenv()

##load rthe groq api
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")


prompt=ChatPromptTemplate.from_template(
    """
answer the questions based on the provided context only.
please provide the most accurate response based on the question 
<context>
{context}
<context>
Question:{input}
"""
)


# def create_vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=OllamaEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## data ingestion
#         st.session_state.docs=st.session_state.loader.load() ## document loading
#         st.session_state.text_spiltter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)\\


def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()

        # ✅ CORRECT NAME: text_splitter (NOT text_spiltter)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # ✅ Use the local variable directly instead of session_state to avoid AttributeError
        st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs[:50])
        
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


user_prompt=st.text_input("enter tour query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("vector database is ready")

import time


if user_prompt:
   document_chain =create_stuff_documents_chain(llm,prompt)
   retriever=st.session_state.vectors.as_retriever()
   retrieval_chain=create_retrieval_chain(retriever,document_chain)


   start=time.process_time()
   response=retrieval_chain.invoke({"input": user_prompt})
   print(f"response time :{ time.process_time()-start}")

   st.write(response["answer"])

   #### streamlit expander 
   with st.expander("Document similarity search"):
       for i,doc in enumerate(response["context"]):
           st.write(doc.page_content)
           st.write("-----------------------")

