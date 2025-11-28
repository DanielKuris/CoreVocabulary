from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained models
sentence_bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('setu4993/LaBSE')
word_bert_model = BertModel.from_pretrained('setu4993/LaBSE')


def sentence_bert_embedding(sentence):
    """Generate sentence embeddings using Sentence-BERT."""
    return [sentence_bert_model.encode(sentence)]


def cosine_similarity_sentences(vec1, vec2):
    """Calculate cosine similarity between two sentence embeddings.
    # Higher semantic similarity should have high cosine similarity (close to 1)"""
    return cosine_similarity(vec1, vec2)[0][0]


# Function to get BERT embeddings for each word in a sentence
def word_bert_embeddings(sentence):
    # Tokenize sentence and get BERT embeddings
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = word_bert_model(**inputs)

    # Extract the hidden states from the last layer
    # outputs[0] -> shape: [batch_size, seq_len, hidden_size]
    embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension

    # Convert to numpy array and ignore [CLS] and [SEP] tokens
    embeddings = embeddings[1:-1].cpu().numpy()

    # Get the list of words (ignoring special tokens like CLS/SEP)
    words = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))[1:-1]

    return words, embeddings


# Function to calculate modified Jaccard similarity using BERT embeddings
def jaccard_similarity_bert(sentence1, sentence2, threshold=0.7):
    # Get words and their embeddings for both sentences
    words1, embeddings1 = word_bert_embeddings(sentence1)
    words2, embeddings2 = word_bert_embeddings(sentence2)

    # Calculate cosine similarities between every pair of words
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    # For each word in sentence1, find the maximum cosine similarity with any word in sentence2
    intersection_similarity = 0
    for i in range(similarity_matrix.shape[0]):
        max_similarity = np.max(similarity_matrix[i])
        intersection_similarity += max_similarity  # Sum up the cosine similarity values

    # Calculate the size of the union (sum of word counts in both sentences)
    union_size = len(words1) + len(words2)

    # Normalize the intersection similarity by dividing it by the union size
    similarity = intersection_similarity / union_size
    return similarity


def compare_sentences(sentence_a, sentence_b):
    cosine_sim_sentences_bert = cosine_similarity_sentences(
        sentence_bert_embedding(sentence_a), sentence_bert_embedding(sentence_b))
    jacard_similarity_bert = jaccard_similarity_bert(sentence_a, sentence_b)

    results = {
        'cosine_similarity_sentences_BERT': cosine_sim_sentences_bert,
        'jaccard_similarity_BERT': jacard_similarity_bert
    }
    return results
