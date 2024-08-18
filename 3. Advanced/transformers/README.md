# Transformer From Scratch 🤖

Welcome to the **Transformer From Scratch** Project! In this notebook, we will implement the transformer model from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Specifically, we will build the transformer encoder-decoder architecture and train it on multiple tasks,

**including:**

- **Text Classification (encoder-only):** We will use the transformer encoder as a feature extractor and add a feedforward network on top to classify text data.
- **Language Modeling (decoder-only):** We will use the transformer decoder to predict the next token in a sequence, which is a fundamental task in natural language processing.
- **Machine Translation (encoder-decoder):** We will combine the transformer encoder and decoder to translate text from one language to another.

By implementing the transformer model from scratch, we'll gain a deep understanding of its architecture, attention mechanism, and multi-head self-attention. We'll also explore how to train the model on different tasks and evaluate its performance.

## Datasets 📚

We will use the following datasets for training and evaluation:

- **IMDb Dataset (Text Classification):** A large movie review dataset containing labeled data for sentiment classification (positive or negative).
- **Wikitext-2 Dataset (Language Modeling):** A collection of Wikipedia articles used for language modeling tasks.
- **Multi30k Dataset (Machine Translation):** A multi-language translation dataset containing image descriptions in English and German.

## Key Techniques 🔧

- **Positional Encoding:** A technique used to inject information about the position of tokens in the input sequence.
- **Multi-Head Self-Attention:** A mechanism that allows the model to focus on different parts of the input sequence simultaneously.
- **Transformer Encoder:** A stack of encoders used to process the input sequence and extract features.
- **Transformer Decoder:** A stack of decoders used to generate the output sequence based on the encoder's features.
- **Feedforward Network:** A neural network used to transform the output of the attention mechanism.

**Let's dive into the code** 📔
