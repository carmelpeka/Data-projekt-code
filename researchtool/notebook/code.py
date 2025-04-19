# Loaders déplacés vers langchain_community
from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Modèle de chat toujours correct
from langchain_community.chat_models import ChatOllama

# Agents : initialize_agent reste, mais load_tools est déplacé
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.agent_types import AgentType

# Math chain, prompts, tools (toujours valables)
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.embeddings import HuggingFaceEmbeddings
# Vector stores et embeddings déplacés
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# Pickle, rien à changer
import pickle

# Chain de QA, pas encore déplacée (à vérifier dans la doc à l'avenir)
from langchain.chains import RetrievalQAWithSourcesChain

# Le package principal (langchain), tu peux l'importer si nécessaire
import langchain
from dotenv import load_dotenv
import streamlit as st
llm = ChatOllama(model="llama3", temperature=0,max_tokens=500)
st.title("Research Tool")
st.sidebar.title("Article Urls")
urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_click=st.sidebar.button("Process URLs")

splitter = RecursiveCharacterTextSplitter(
    # separators=["\n\n", "\n", " "],
    chunk_size=1000,
    chunk_overlap=200

)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
main_placefolder=st.empty()
if process_url_click:
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading..start..")
    data=loader.load()
    docs = splitter.split_documents(data)
    main_placefolder.text("Data splitting..start..")
    vertorindex = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Data vertorize..start..")
    vertorindex.save_local("my_faiss_index")

query=main_placefolder.text_input("question: ")
if query:
    vectorindex = FAISS.load_local("my_faiss_index", embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex.as_retriever(),return_source_documents=True)
    result=chain.invoke({"question":query},return_only_outputs=True)
    st.header("Answer")
    st.subheader(result["answer"])
    sources=result['source_documents'][1].metadata["source"]
    st.write(sources)