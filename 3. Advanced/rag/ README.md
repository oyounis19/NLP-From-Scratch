# RAG (Retreival Augmented Generation) 🤖

Welcome to the RAG (Retrieval Augmented Generation) Project! In this project, we will implement a multimodal RAG system that allows users to upload and retrieve information from text documents, images, videos, and audio files. The system uses dense retrieval techniques with embeddings to enable highly efficient search and generation of relevant content.

## Project Overview 📚

This project showcases how to build a RAG system from scratch, where **users can upload data of type (PDF) and ask questions based on the contents**. The system extracts relevant information from all types of media, using state-of-the-art models for text, image, and audio processing. The system then uses a generative model to synthesize a response based on the retrieved information.

## Key Techniques 🔧

This project leverages a variety of powerful machine learning tools and models to implement a fully functional RAG system:

- Sentence-BERT: For extracting embeddings from text.
- **FAISS (Facebook AI Similarity Search):** For efficient vector search and retrieval.
- **PyTorch:** The backbone deep learning library for implementing all models.
- **Hugging Face Transformers:** Pre-trained models for text generation and fine-tuning.
- **Few-Shot Prompting:** to instruct the model on how to generate responses.

## Models Used 🤖

The RAG system uses the following models to perform various tasks:

- **all-mpnet-base-v2:** A pre-trained model for extracting embeddings from text.
- **microsoft/Phi-3.5-mini-instruct:** A llm model for text generation.

**Let's dive into the notebook** 📔
