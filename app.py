import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import trafilatura
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PDF_FOLDER = "data"
WEBSITE_URL = "https://www.angelone.in/support"

# Page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 Customer Support Chatbot")
st.markdown("This chatbot answers only from built-in support documents and website.")

# === API Keys ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set your Google API key in `.env`")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
CHAT_MODEL = 'gemini-1.5-flash'
EMBEDDING_MODEL = 'models/embedding-001'

# === Prompt Template ===
custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use only the following context to answer the question.
If the answer is not in the context, respond with "I don't know".

Context:
{context}

Question: {question}
""")


def fetch_website_content(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text or ""
    except Exception as e:
        st.warning(f"Failed to fetch {url}: {e}")
        return ""


def load_all_pdfs_from_folder(folder_path):
    all_docs = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    all_docs += loader.load()
                except Exception as e:
                    st.warning(f"Failed to load PDF {filename}: {e}")
    else:
        st.warning(f"PDF folder {folder_path} does not exist. Creating it now.")
        os.makedirs(folder_path, exist_ok=True)
    return all_docs


@st.cache_resource(show_spinner=True)
def create_knowledge_base(pdf_folder, website_url):
    # Load PDFs
    pdf_docs = load_all_pdfs_from_folder(pdf_folder)

    # Website content
    site_text = fetch_website_content(website_url)
    website_docs = []
    if site_text:
        website_doc = Document(page_content=site_text, metadata={"source": website_url})
        website_docs = [website_doc]

    # Combine all documents
    all_docs = pdf_docs + website_docs

    if not all_docs:
        st.warning("No documents found. Please add PDFs to the data folder or check website access.")
        # Create a minimal document to prevent errors
        empty_doc = Document(page_content="No content available", metadata={"source": "empty"})
        all_docs = [empty_doc]

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # Use Google's embeddings instead
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


# === Build Knowledge Base ===
try:
    with st.spinner("Building knowledge base... This may take a minute."):
        vector_store = create_knowledge_base(PDF_FOLDER, WEBSITE_URL)
    retriever = vector_store.as_retriever(search_type="similarity", k=4)
except Exception as e:
    st.error(f"Error creating knowledge base: {e}")
    st.stop()


# === QA Chain ===
def get_gemini_response(question, context):
    """
    Retrieves answer from Gemini
    """
    try:
        model = genai.GenerativeModel(CHAT_MODEL)
        prompt = custom_prompt.format(context=context, question=question)
        response = model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else "I don't know"
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Sorry, I couldn't generate a response due to an error: {e}"


def augment_context_and_query(question, retriever):
    """
    Fetches relevant contexts and combines them with the query.
    """
    try:
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])
        return context_text
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return "No context available due to an error."


# === Chat Session ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Chat Input + Display ===
user_input = st.chat_input("Ask your question...")
if user_input:
    with st.spinner("Thinking..."):
        context = augment_context_and_query(user_input, retriever)
        answer = get_gemini_response(user_input, context)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", answer))

# Show conversation
for sender, msg in st.session_state.chat_history:
    st.chat_message(sender).write(msg)