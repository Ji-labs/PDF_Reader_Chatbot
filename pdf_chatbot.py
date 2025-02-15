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
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.9, top_p=0.95)
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template="""You are a helpful AI assistant that helps users understand their PDF documents.
        Use ALL available context to provide a comprehensive answer.

        {context}

        Question: {question}
        Comprehensive Answer:"""
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt, 'max_tokens_limit': 4096}
    )
    return conversation_chain

def process_docs(pdf_docs):
    try:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        assert len(embeddings.embed_query("test")) == 768, "Embedding dimension mismatch!"
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True
        return True
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing PDFs..."):
            if process_docs(pdf_docs):
                st.success("Processing complete!")

if st.session_state.processComplete:
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": user_question})
                st.write("Retrieved Context:", response.get('source_documents', 'No context found'))
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"Error: {str(e)}")
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.write("👈 Upload PDFs to start chatting!")
