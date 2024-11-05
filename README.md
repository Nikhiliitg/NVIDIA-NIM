# NVIDIA NIM RAG Application

This application leverages NVIDIA's NIM (Neural Inference Model) to provide a Retrieval-Augmented Generation (RAG) system that allows users to ask questions based on the context provided by PDF documents. The app uses LangChain for document processing and retrieval and Streamlit for a user-friendly web interface.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
  
## Features
- Upload and process PDF documents from a specified directory.
- Create embeddings using NVIDIA’s AI endpoints.
- Answer user queries based on the context derived from the loaded documents.
- Measure and display response times for queries.

## Requirements
To run this application, you need the following:
- Python 3.7 or higher
- Streamlit
- LangChain
- NVIDIA API access
- Required libraries in `requirements.txt`

## Installation
1. Clone the repository:

   git clone <https://github.com/Nikhiliitg/NVIDIA-NIM.git>
   cd <https://github.com/Nikhiliitg/NVIDIA-NIM.git>

2. Install the required packages:
    pip install -r requirements.txt

3. Set up your NVIDIA API credentials and configuration.
    NVIDIA_API_KEY=your_nvidia_api_key


## Usage

1. Run the Streamlit app:

   streamlit run finalapp.py
   
Open your web browser and go to http://localhost:8501 to interact with the app.
Use the input box to enter your questions based on the provided PDF documents. Click "Document Embedding" to initialize the vector store, then ask your questions to receive answers.
How It Works

The application loads PDF documents from the specified directory and uses the NVIDIAEmbeddings class to create vector embeddings.
User queries are processed using the ChatNVIDIA model to generate responses based on the context of the documents.
FAISS (Facebook AI Similarity Search) is used for efficient document retrieval.
Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

