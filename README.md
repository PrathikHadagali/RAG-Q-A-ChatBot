# RAG Q&A ChatBot using Groq and Llama3

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot using Groq's Llama3 model and FAISS vector database for efficient document retrieval. The chatbot answers user queries based on research papers stored in a vector database.

## Features

- Uses LangChain for document retrieval and processing
- Embeds research papers using OllamaEmbeddings or OpenAIEmbeddings for Fast Processing.
- Loads PDFs from a directory using PyPDFDirectoryLoader
- Splits documents into manageable chunks using RecursiveCharacterTextSplitter
- Stores embeddings in a FAISS vector database
- Uses ChatGroq as the LLM backend
- Provides document similarity search along with answers
- Streamlit UI for easy interaction.

## Installation
### Prerequisites
- Ensure you have Python installed. Install the required dependencies using:
  pip install streamlit langchain langchain_groq langchain_community python-dotenv faiss-cpu

## Setup
## 1.Clone the repository:
[git clone https://github.com/your-username/your-repo.git](https://github.com/PrathikHadagali/RAG-Q-A-ChatBot.git)
## 2.Set up environment variables:
Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY=your_api_key_here
## 3.Place research papers inside the research_papers directory.

## Usage
- Run the Streamlit application:
- streamlit run app.py
  
## Workflow
- Click "Document Embeddings" to process the research papers.
- Enter a query in the text input field.
- Click enter to get an answer based on the documents.
- View document similarity results under the "Document Similarity Search" expander.

## Project Structure
├── research_papers/       # Directory containing PDF research papers
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (not included in repo)
├── README.md              # Project documentation

# Author
Prathik M Hadagali - [GitHub Profile](https://github.com/PrathikHadagali/RAG-Q-A-ChatBot)
