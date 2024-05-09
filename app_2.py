import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PIL import Image

# Load environment variables
load_dotenv()

# Define functions for the chatbot

def fetch_article_title(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else "No title found"
    except requests.RequestException:
        return "Failed to fetch title."

def display_article_title(title):
    st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <h2 style="font-weight:bold; color:black;">{title}</h2>
        </div>
    """, unsafe_allow_html=True)


def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']



# Application UI
logo_url = "https://i.pinimg.com/originals/3e/16/04/3e1604c9c848ec7479ae3626066b78d2.jpg"

background_url = "https://getwallpapers.com/wallpaper/full/2/d/4/101359.jpg"
css = f"""
<style>
.stApp {{
    background-image: url({background_url});
    background-size: cover;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

#st.title("Chat with The Bucky")

st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px;">
        <img src="{logo_url}" width="100"> 
        <h1 style="margin:0;">Chat with Bucky</h1>
    </div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL", key="website_url_input")

if website_url:
    if "article_title" not in st.session_state:
        st.session_state.article_title = fetch_article_title(website_url)
    display_article_title(st.session_state.article_title)
    if "vector_store" not in st.session_state or st.session_state.website_url != website_url:
        st.session_state.website_url = website_url
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am Buckey. How can I help you?")]

user_query = st.text_input("Type your message here...", key="user_query_input")
if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)