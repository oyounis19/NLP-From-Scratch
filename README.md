# NLP-From-Scratch 🚀📚

![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Welcome to **NLP-From-Scratch**! 🌟 This is your ultimate playground for diving into the world of Natural Language Processing (NLP) with PyTorch. Whether you're just getting started or looking to tackle advanced projects, this repository has something for everyone. Let's embark on this exciting journey to master NLP techniques together! 🤖✨

## Table of Contents 📖

1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Projects](#projects)
5. [Advanced Language Model](#advanced-language-model)
6. [Contributing](#contributing)
7. [Acknowledgements](#acknowledgements)

## Introduction

Natural Language Processing (NLP) is a fascinating field of artificial intelligence that enables machines to understand, interpret, and generate human language. This repository is designed to demystify NLP through hands-on projects that guide you through various tasks and techniques, all implemented from scratch! 🛠️💡

## Repository Structure

The repository is organized into three levels, each offering a set of projects to help you build your NLP skills step-by-step:

1. **Basics**: This level covers fundamental NLP concepts and techniques, such as text preprocessing, word embeddings, text classification, and more.

2. **Intermediate**: This level delves into more advanced NLP tasks, including part-of-speech tagging, named entity recognition, question answering, and sentiment analysis with recurrent neural networks (RNNs).

3. **Advanced**: This level explores cutting-edge NLP models and architectures, such as transformers, BERT, GPT, neural machine translation, and multi-task learning.

Each project is self-contained and includes detailed explanations, code examples, and references to help you understand and implement the concepts effectively. 📚🚀

## Getting Started

Let's set up your environment so you can start experimenting with the projects! 💻🧪

### Prerequisites

- Python 3.x 🐍
- PyTorch 🔥
- Jupyter Notebook (optional, for interactive exploration) 📓

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/NLP-From-Scratch.git
cd NLP-From-Scratch
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

You're all set! 🎉

## Projects

1. **Basics** 🌱
    - [**Text Preprocessing: 🧹**](1.%20Basics/text-preprocessing)  
    Learn essential text preprocessing techniques such as tokenization, stemming, lemmatization, stop word removal, and text normalization.
        - **Objective:** Transform raw text into a format suitable for NLP tasks.
        - **Key Techniques:** Tokenization, Stopword Removal, Stemming, Lemmatization.
        - **Tools:** NLTK, spaCy.

    - [**Word Embeddings: 🧠**](1.%20Basics/word-embeddings)  
    Explore word embeddings and their applications in NLP, including word2vec, GloVe, and fastText.
        - **Objective:** Represent words as dense vectors to capture semantic relationships.
        - **Key Techniques:** Word2Vec, GloVe, fastText.
        - **Tools:** Gensim, spaCy.
    - [**Text Classification: 📊**](1.%20Basics/text-classification)  
    Build a text classifier using machine learning algorithms and evaluate its performance on a dataset.
        - **Objective:** Classify text documents into predefined categories.
        - **Key Techniques:** Bag-of-Words, TF-IDF, Naive Bayes, Logistic Regression.
        - **Tools:** scikit-learn.
    - [**Bag of Words vs. TF-IDF 🔍**](1.%20Basics/bow-vs-tfidf)  
    Compare the Bag-of-Words and TF-IDF approaches for text representation and classification.
        - **Objective:** Understand the differences between Bag-of-Words and TF-IDF.
        - **Key Techniques:** Bag-of-Words, TF-IDF.
        - **Tools:** scikit-learn.
2. **Intermediate** 🌿
    - [**Part-of-Speech Tagging: 🏷️**](2.%20Intermediate/part-of-speech-tagging)  
    Implement a part-of-speech tagger using Hidden Markov Models (HMMs) and the Viterbi algorithm.
        - **Objective:** Assign part-of-speech tags to words in a sentence.
        - **Key Techniques:** Hidden Markov Models, Viterbi Algorithm.
        - **Tools:** NLTK.
    - [**Named Entity Recognition: 🏷️**](2.%20Intermediate/named-entity-recognition)  
    Develop a named entity recognition (NER) system using conditional random fields (CRFs) and evaluate its performance.
        - **Objective:** Identify named entities (e.g., persons, organizations, locations) in text.
        - **Key Techniques:** Conditional Random Fields (CRFs).
        - **Tools:** scikit-learn, CRFsuite.
    - [**Question Answering: ❓**](2.%20Intermediate/question-answering)  
    Create a question answering system using a simple heuristic approach and evaluate its effectiveness.
        - **Objective:** Generate answers to questions based on a given context.
        - **Key Techniques:** Heuristic Search, Text Similarity.
        - **Tools:** spaCy.
    - [**Sentiment Analysis with RNNs, LSTMs, GRUs, and CNNs: 📈**](2.%20Intermediate/sentiment-analysis)  
    Perform sentiment analysis on text data using recurrent neural networks (RNNs), long short-term memory (LSTM) networks, gated recurrent units (GRUs), and convolutional neural networks (CNNs).
        - **Objective:** Predict the sentiment (positive, negative, neutral) of text data.
        - **Key Techniques:** RNNs, LSTMs, GRUs, CNNs.
        - **Tools:** PyTorch.

3. **Advanced** 🚀
    - [**Transformers: 🤖**](3.%20Advanced/transformers)  
    Implement the transformer architecture from scratch and explore its applications.
        - **Objective:** Understand the architecture and working of transformer models.
        - **Key Techniques:** Self-Attention Mechanism, Positional Encoding, Multi-Head Attention.
        - **Tools:** Pytorch.
    - [**BERT: 🤗**](3.%20Advanced/bert)  
    Explore BERT (Bidirectional Encoder Representations from Transformers), one of the most popular transformer models, and fine-tune a pre-trained BERT model for text classification.
        - **Objective:** Fine-tune a pre-trained BERT model for text classification tasks.
        - **Key Techniques:** BERT, Tokenization, Attention Mechanism, Transfer Learning.
        - **Tools:** Hugging Face Transformers, PyTorch.
    - [**GPT: 🧠**](3.%20Advanced/gpt)  
    Discover GPT (Generative Pre-trained Transformer), a state-of-the-art language model, and generate text using a pre-trained GPT model.
        - **Objective:** Generate text using a pre-trained GPT model.
        - **Key Techniques:** GPT, Autoregressive Language Modeling.
        - **Tools:** Hugging Face Transformers, PyTorch.
    - [**Neural Machine Translation: 🌍**](3.%20Advanced/neural-machine-translation)  
    Build a neural machine translation (NMT) system using an encoder-decoder architecture with attention mechanism.
        - **Objective:** Translate text from one language to another using neural networks.
        - **Key Techniques:** Encoder-Decoder Architecture, Attention Mechanism.
        - **Tools:** PyTorch.
    - [**Multi-Task Learning: 🎓**](3.%20Advanced/multi-task-learning)  
    Implement a multi-task learning model that jointly learns multiple NLP tasks, such as part-of-speech tagging, named entity recognition, and text classification.
        - **Objective:** Train a single model to perform multiple NLP tasks simultaneously.
        - **Key Techniques:** Multi-Task Learning, Shared Representations.
        - **Tools:** PyTorch.

## Advanced Language Model

In addition to this repository, we have another project dedicated to building a small language model (LLM) from scratch. This comprehensive project covers:

- **Building an LLM:** Learn the intricacies of designing and implementing your own language model.
- **Pretraining:** Train your model on a large corpus to grasp general language understanding.
- **Finetuning:** Adapt the model for specific tasks like sentiment analysis, named entity recognition, and more.
- **Integration with RAG:** Implement a Retrieval-Augmented Generation (RAG) system to enhance the model's capabilities using external knowledge.

You can find the detailed project [here](https://github.com/oyounis19/LLM-from-scratch). 🌐✨

## Contributing

Contributions are welcome! If you'd like to contribute to this repository, please follow these steps:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-improvement`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-improvement`).
6. Create a new Pull Request.

If you find any issues or have suggestions for new projects, feel free to open an issue or submit a pull request. Let's learn and grow together! 🌟

## Acknowledgements

<p style="text-align: center; font-family: 'Georgia', serif; font-size: 24px; color: #fff; margin-top: 40px;">
  <em>As always, AI is just statistics on steroids :)</em>
</p>
