import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def read_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunk_data(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=250, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text):
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text, embedding=embeddings)
    return vectorstore

def get_conversation(vectorstore):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handleInput(user_text, conversation_chain):
    res = conversation_chain({'question': user_text})
    chat_history = res['chat_history']
    ans =  res['answer'] 
    st.write(ans)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Chat with PDF")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat With PDF")

    user_text = st.text_input("Ask question:")
    if user_text and st.session_state.conversation:
        handleInput(user_text, st.session_state.conversation)
        
    with st.sidebar:
        st.subheader("Your Documents")
        pdf = st.file_uploader("Upload PDF")
        if pdf and st.button("Submit"):
            with st.spinner("Processing..."):
                # Read data from pdf
                raw_text = read_pdf(pdf)
                # Split data into chunks
                load_chunks = get_chunk_data(raw_text)
                # Create a vector store 
                vector_store = get_vector_store(load_chunks)
                
                # Create conversation chain
                conversation_chain = get_conversation(vector_store)
                st.session_state.conversation = conversation_chain  # Save the conversation chain to session state

if __name__ == '__main__':
    main()
