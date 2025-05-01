 RAG Chatbot for Customer Support

A Retrieval-Augmented Generation (RAG) chatbot that answers customer support questions based on your own knowledge base of PDF documents and website content.

![Chatbot Screenshot](https://via.placeholder.com/800x450?text=RAG+Chatbot+Screenshot)

 Features

PDF Document Processing: Automatically processes PDF files in the data folder
Website Integration: Scrapes and processes content from a support website
Vector Search: Uses FAISS for fast similarity searching
Contextual Responses: Generates answers based only on your knowledge base
Streamlit UI: Clean, interactive web interface

 How It Works

This app implements a RAG (Retrieval-Augmented Generation) pipeline:

1. Indexing: PDF documents and website content are processed, chunked, and indexed
2. Retrieval: When a question is asked, relevant content is retrieved from the knowledge base
3. Generation: The retrieved context and question are sent to Google's Gemini model
4. Response: A relevant answer is provided based only on the retrieved information

 Installation

 Prerequisites

Python 3.8+
Google API key for Gemini

 Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rag-customer-support.git
   cd rag-customer-support
   ```

2. Install required packages:
   ```bash
   pip install streamlit google-generativeai python-dotenv trafilatura
   pip install langchain langchain-community langchain-core langchain-text-splitters langchain-google-genai
   pip install faiss-cpu pypdf
   ```

3. Create a `.env` file in the project root with your API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Create a `data` folder in the project root:
   ```bash
   mkdir data
   ```

5. Add your PDF documents to the `data` folder.

 Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Start asking questions about your documents!

 Configuration

You can customize the following variables in the script:

`PDF_FOLDER`: Location of your PDF documents (default: "data")
`WEBSITE_URL`: Support website to include in knowledge base
`CHAT_MODEL`: Gemini model for chat responses (default: "gemini-1.5-flash")
`EMBEDDING_MODEL`: Google's embedding model (default: "models/embedding-001")

 How to Add Your Knowledge Base

 Adding PDFs

Simply place PDF files in the `data` folder. The application will automatically process them.

 Changing Website Source

To use a different website as part of your knowledge base, update the `WEBSITE_URL` variable:

```python
WEBSITE_URL = "https://your-support-website.com/faq"
```

 Customization

 Modifying the Prompt

You can customize how the chatbot responds by editing the prompt template:

```python
custom_prompt = PromptTemplate.from_template("""
You are a helpful customer support assistant for our company.
Use only the following context to answer the question.
If the answer is not in the context, respond with "I don't have that information.
Would you like me to connect you with a human agent?"

Context:
{context}

Question: {question}
""")
```

 Adjusting Chunk Size

For different document types, you might want to adjust how content is chunked:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

 Troubleshooting

 API Key Issues

If you see an error about the API key:
Make sure the `.env` file exists and contains your valid API key
Check that `load_dotenv()` is called before accessing environment variables

 PDF Loading Issues

If PDFs aren't loading:
Verify the PDF files are in the `data` folder
Check if the PDFs are accessible and not corrupted
Make sure you have the correct permissions to read the files

 Website Scraping Issues

If website content isn't being fetched:
Check your internet connection
Verify the website allows scraping
The website might have anti-scraping measures in place

 Limitations

The chatbot only answers based on the provided documents and website
Large PDF files may take longer to process
Some websites might block web scraping

 Future Improvements

Add support for more document types (DOCX, TXT, etc.)
Implement conversation memory
Add feedback mechanism for responses
Include source citations in responses
Add authentication for secure deployment

 License

[MIT License](LICENSE)

 Acknowledgments

[LangChain](https://github.com/hwchase17/langchain) for the RAG framework
[Streamlit](https://streamlit.io/) for the web interface
[Google Generative AI](https://ai.google.dev/) for the language models
[FAISS](https://github.com/facebookresearch/faiss) for vector searching
[Trafilatura](https://github.com/adbar/trafilatura) for web scraping
