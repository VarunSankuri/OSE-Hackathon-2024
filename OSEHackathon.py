import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PyPDF2
import os
import io
from langchain_community.vectorstores import Chroma
import pysqlite3  # Add this import
import sys       # Add this import

# Swap sqlite3 with pysqlite3-binary
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

st.title("OSE Hackathon")
st.markdown("""
<style>
.big-font {
  font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)


# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
# google_api_key = os.getenv("GOOGLE_API_KEY")
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

tab1, tab2 = st.tabs(
    [ "Chat Bot","Upload PDF Files"]
)

with tab2:
    st.caption("Although not necessary, you can upload your PDFs here to get more accurate answers/code")
    # File Upload with multiple file selection
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.text("PDF Files Uploaded Successfully!")

        # Combine all PDF content
        all_texts = []
        for uploaded_file in uploaded_files:
            # PDF Processing
            pdf_data = uploaded_file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            pdf_pages = pdf_reader.pages

            # Extract text from all pages and add to the combined context
            context = "\n\n".join(page.extract_text() for page in pdf_pages)
            all_texts.append(context)

        # Combine all contexts into a single string
        combined_context = "\n\n".join(all_texts)

        # Split Texts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(combined_context)

        # Chroma Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

with tab1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if question := st.chat_input("Ask your Cloud related questions here. For e.g. AWS Cognito v/s Google Firebase"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Get Relevant Documents (only if files were uploaded)
        if uploaded_files:
            docs = vector_index.get_relevant_documents(question)
        else:
            docs = []  # No documents to provide

        # Define Prompt Template
        prompt_template = """
        You are a helpful AI assistant helping people answer their Cloud development and
        deployment questions. Answer the question as detailed as possible from the provided context,
        make sure to provide all the details and code if possible, if the answer is not in
        provided context use your  knowledge or imagine an answer but never say that you don't have an answer
        or can't provide an answer based on current context ",

        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """

        # Create Prompt
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

        # Load QA Chain
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1, api_key=google_api_key)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Get Response
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['output_text']})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response['output_text'])
