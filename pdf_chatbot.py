import streamlit as st 
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Chat with PDF", page_icon="📚")
st.title("Chat with your PDF 📚")

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.7)
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template="""You are an AI assistant providing detailed answers from uploaded PDF documents.
        Use all provided context to generate accurate responses.

        {context}

        Question: {question}
        Answer:"""
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={'prompt': prompt})
    return conversation_chain

def process_docs(pdf_docs):
    try:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

with st.sidebar:
    st.subheader("Upload your PDF Documents")
    pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing PDFs..."):
            if process_docs(pdf_docs):
                st.success("PDFs processed successfully!")

if st.session_state.processComplete:
    user_question = st.chat_input("Ask something about your PDFs:")
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": user_question})
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"Error: {str(e)}")
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.write("👈 Upload PDFs to begin chatting!")
