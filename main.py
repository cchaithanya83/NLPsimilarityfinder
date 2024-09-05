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

# Function to split a text into chunks of specified size
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
input_string = """Artificial intelligence (AI) is a field of computer science that focuses on creating machines capable of intelligent behavior. It has been defined in many ways, but in general, it involves the simulation of human intelligence by machines. Over the years, AI has evolved significantly, leading to the development of machine learning, deep learning, natural language processing, and computer vision. AI technologies have been integrated into various applications, ranging from chatbots and virtual assistants to autonomous vehicles and predictive analytics. One of the most significant breakthroughs in AI research was the development of deep learning algorithms, which are inspired by the structure and function of the human brain. These algorithms use artificial neural networks to learn from large amounts of data and improve their performance over time.

Machine learning, a subset of AI, involves the use of algorithms that enable computers to learn from and make predictions or decisions based on data. Unlike traditional programming, where rules are explicitly defined, machine learning allows the model to discover patterns and relationships within the data. There are several types of machine learning, including supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, while unsupervised learning deals with unlabeled data to identify hidden patterns. Reinforcement learning, on the other hand, involves training models to make sequences of decisions by rewarding them for desired outcomes.

Natural language processing (NLP) is another critical area of AI that focuses on enabling machines to understand, interpret, and generate human language. NLP has numerous applications, such as language translation, sentiment analysis, and text summarization. With the advent of transformer-based models like BERT, GPT, and T5, the field of NLP has witnessed remarkable advancements. These models use deep learning techniques to capture the context and meaning of words in a sentence, leading to more accurate and nuanced language understanding.

Computer vision, a field related to AI, is concerned with enabling machines to interpret and understand visual information from the world. Applications of computer vision include facial recognition, object detection, and image segmentation. By using convolutional neural networks (CNNs), computer vision systems can analyze and classify images and videos with high accuracy.

The impact of AI on society is profound, with potential benefits ranging from improved healthcare and education to enhanced decision-making in businesses. However, it also raises ethical concerns, such as privacy issues, job displacement, and the potential for biased decision-making. Researchers and policymakers are actively working to address these challenges by developing fair and transparent AI systems and establishing ethical guidelines."""
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

# After calculating similarities
# Find the chunk with the highest cosine similarity
most_similar_chunk, highest_similarity = max(similarities, key=lambda x: x[1])

print(f"Most Similar Chunk: {most_similar_chunk}\nHighest Cosine Similarity: {highest_similarity:.4f}\n")
