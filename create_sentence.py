import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from TransformersUtilityFunctions import get_vec
import pickle as pkl
from sentence_similarity import compare_sentences

# Load NLTK stop words
stop_words = set(stopwords.words('english'))

# Current vocabulary
with open('vocab_words_formatted.txt', 'r') as file:
    curr_vocab = eval(file.read())


# Step 1: Preprocess the sentence by tokenizing, handling stop words, and identifying out-of-vocabulary words
def preprocess_sentence(sentence, vocab):
    # Tokenize sentence into words
    words = sentence.split()

    # Remove stop words only if they are NOT in the limited vocabulary
    processed_words = [
        word for word in words
        if word.lower() not in stop_words or word in vocab
    ]

    # Check which words are out of vocabulary
    out_of_vocab_words = [word for word in processed_words if word not in vocab]

    return processed_words, out_of_vocab_words


# Step 2: Find the closest word in the limited vocabulary using cosine similarity
def find_closest_word(word, vocab_embeddings, word_embeddings):
    if word not in word_embeddings:
        return None  # Handle if the word has no embedding

    word_embedding = word_embeddings[word].reshape(1, -1)  # Reshape to (1, n) for cosine similarity

    # Extract embeddings from the vocabulary embeddings dictionary
    vocab_embedding_list = np.array(list(vocab_embeddings.values()))

    # Find the most similar word from the allowed vocabulary
    # Ensure both embeddings are 2D arrays (reshape if necessary)
    word_embedding_2d = word_embedding.reshape(1, -1)  # Convert to 2D array if it's 1D
    vocab_embeddings_2d = np.vstack(list(vocab_embeddings.values()))  # Stack vocab embeddings into a 2D array

    similarities = cosine_similarity(word_embedding_2d, vocab_embeddings_2d)

    # Find the index of the most similar word
    closest_idx = np.argmax(similarities)

    # Return the word corresponding to the closest_idx
    closest_word = list(vocab_embeddings.keys())[closest_idx]

    return closest_word


# Step 3: Reconstruct the sentence using substitutions
def reconstruct_sentence(original_words, substitutions):
    # Rebuild the sentence with substitutions
    transformed_sentence = [
        substitutions.get(word, word) for word in original_words
    ]

    return " ".join(transformed_sentence)


# Step 5: Transform the sentence by combining all steps
def transform_sentence(sentence, vocab, vocab_embeddings, word_embeddings):
    # Step 1: Preprocess
    original_words, out_of_vocab_words = preprocess_sentence(sentence, vocab)

    # Step 2: Substitute words
    substitutions = {}
    for word in out_of_vocab_words:
        closest_word = find_closest_word(word, vocab_embeddings, word_embeddings)
        if closest_word:
            substitutions[word] = closest_word

    # Step 3: Reconstruct the sentence
    transformed_sentence = reconstruct_sentence(original_words, substitutions)

    return transformed_sentence


# Populate embeddings for the limited vocabulary
def build_vocab_embeddings(vocab):
    vocab_embeddings = {}
    for word in vocab:
        vec = get_vec(word)
        if vec is not None:  # Only add the embedding if the word has an embedding
            print(f'{len(vocab_embeddings)}/{len(vocab)}: {word}')
            vocab_embeddings[word] = vec
    return vocab_embeddings


# Populate embeddings for the entire vocabulary (words that can appear in input sentences)
def build_word_embeddings(sentence):
    words = set(sentence.split())  # Get all unique words in the sentence
    word_embeddings = {}
    for word in words:
        vec = get_vec(word)
        if vec is not None:  # Only add the embedding if the word has an embedding
            # print(f'{len(word_embeddings)}/{len(words)}: {word}')
            word_embeddings[word] = vec
    return word_embeddings


def similarity_checker(original_sentence, transformed_sentence):
    similarities = compare_sentences(original_sentence, transformed_sentence)
    print(f"Cosine Similarity (Sentence-BERT): {similarities['cosine_similarity_sentences_BERT']}")
    print(f"jaccard Similarity (Sentence-BERT): {similarities['jaccard_similarity_BERT']}")

def create_sentence(original_sentence, embbedings_path="vocab_embeddings_dict.pkl"):
    # vocab_embeddings = build_vocab_embeddings(curr_vocab)
    with open(embbedings_path, 'rb') as file:
        vocab_embeddings = pkl.load(file)

    word_embeddings = build_word_embeddings(original_sentence)

    return transform_sentence(original_sentence, curr_vocab, vocab_embeddings, word_embeddings)


# Example usage
if __name__ == "__main__":
    sentence = "The kid had a gun"

    vocabulary = curr_vocab

    with open("vocab_embeddings_dict.pkl", 'rb') as file:
        vocab_embeddings = pkl.load(file)
    print(f'vocab_embeddings.....DONE!')
    word_embeddings = build_word_embeddings(sentence)
    print(f'word_embeddings.....DONE!')

    # Transform the sentence
    transformed_sentence = transform_sentence(sentence, vocabulary, vocab_embeddings, word_embeddings)

    print("Original Sentence:", sentence)
    print("Transformed Sentence:", transformed_sentence)
    similarity_checker(sentence, transformed_sentence)
