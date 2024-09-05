# NLP Chunk Similarity Finder

This project provides a way to find the most relevant chunks of text based on a question by calculating the cosine similarity between embeddings generated using a pre-trained transformer model. The model used in this project is `roberta-large`, which is part of the Hugging Face Transformers library.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Example](#example)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project utilizes the `transformers` library to load a pre-trained language model (`roberta-large`) and the `sklearn` library to calculate cosine similarity between embeddings. Given a long text and a question, it splits the text into chunks based on sentences and computes the cosine similarity between the question's embedding and each chunk's embedding to find the most relevant chunk.

## Setup

To set up this project, you need to have Python installed. Follow these steps to set up the environment:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/nlp-chunk-similarity-finder.git
    cd nlp-chunk-similarity-finder
    ```

2. Install the required libraries:

    ```bash
    pip install torch transformers scikit-learn
    ```

## Usage

1. **Load the Pre-trained Model and Tokenizer:** The script initializes the `roberta-large` model and tokenizer from Hugging Face.

2. **Define the Text and Question:** Provide a long text input and a question. The script splits the text into chunks based on sentences.

3. **Compute Embeddings and Cosine Similarity:** The script computes embeddings for the question and each chunk of text. It then calculates the cosine similarity between the question's embedding and each chunk's embedding.

4. **Get the Most Similar Chunk:** The script outputs the chunk with the highest similarity score to the question, indicating the most relevant chunk of text.

## Example

Here's how you can use the script:

```python
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model and tokenizer
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings for a given text
def get_embedding(text):
    # Tokenize the input text and get the hidden states from the model
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token's embedding as the sentence-level embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return cls_embedding

# Function to split a text into chunks based on '.'
def chunk_text_by_sentence(text):
    # Split text by '.'
    sentences = text.split('.')
    # Remove any extra spaces and filter out empty sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

# Function to calculate cosine similarity between two embeddings
def cosine_similarity_between_embeddings(embedding1, embedding2):
    # Reshape the embeddings for sklearn's cosine_similarity function
    embedding1 = embedding1.numpy().reshape(1, -1)
    embedding2 = embedding2.numpy().reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

# Example input string and question
input_string = """..."""  # Long input text
question = "What are the applications of natural language processing (NLP)?"

# Get embedding for the question
question_embedding = get_embedding(question)

# Split the input string into chunks
chunks = chunk_text_by_sentence(input_string)

# Calculate cosine similarity between the question and each chunk
similarities = []
for i, chunk in enumerate(chunks):
    chunk_embedding = get_embedding(chunk)
    similarity = cosine_similarity_between_embeddings(question_embedding, chunk_embedding)
    similarities.append((chunk, similarity))

# Print the similarities for each chunk
for chunk, similarity in similarities:
    print(f"Chunk: {chunk}\nCosine Similarity: {similarity:.4f}\n")

# Find the chunk with the highest cosine similarity
most_similar_chunk, highest_similarity = max(similarities, key=lambda x: x[1])
print(f"Most Similar Chunk: {most_similar_chunk}\nHighest Cosine Similarity: {highest_similarity:.4f}\n")
```

Replace `...` with your input text.

## Requirements

- Python 3.7+
- `torch` - PyTorch library for deep learning.
- `transformers` - Hugging Face Transformers library.
- `scikit-learn` - Machine learning library for Python.

Install the requirements using:

```bash
pip install torch transformers scikit-learn
```

