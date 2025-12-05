import nltk
from nltk.corpus import brown, stopwords, wordnet as wn
from collections import Counter
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import sys
import runpy
import pandas as pd
import csreparser

voc_size = int(sys.argv[1])


# Download necessary NLTK data
nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
print("Done downloading!")


# Step 1: Initial Vocabulary Extraction
def extract_vocabulary(corpus, corpus_size=10000):
    # corpus = brown.words()
    # convert to lower case latter AND filter non-alphabetic characters
    words = [word.lower() for word in corpus if word.isalpha()]
    # filter stopwords (i.e. 'i', 'me', 'am', 'are', 'because', etc...)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # count the requency of each word and select the most common words
    word_counts = Counter(words)
    most_common_words = [word for word, freq in word_counts.most_common(corpus_size)]
    return most_common_words


# Step 2: Semantic Coverage with WordNet
def get_hypernyms(word):
    # find "hypernym" (more general term) for each word
    synsets = wn.synsets(word)
    hypernyms = set()
    for synset in synsets:
        hypernyms.update(lemma.name() for hypernym in synset.hypernyms() for lemma in hypernym.lemmas())
    return list(hypernyms)


def create_basic_words(most_common_words, vocabulary_size=300):
    basic_words = set()
    for word in most_common_words:
        basic_words.update(get_hypernyms(word))
    return list(basic_words)  # [:vocabulary_size]  # Limit to 600 words


# Step 3: Filtering with Embeddings (GloVe)
def load_glove_embeddings(glove_file):
    model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    return model


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def filter_vocabulary(vocabulary, model, threshold=0.7):
    # Precompute word embeddings for all words in the vocabulary
    word_embeddings = []
    valid_words = []

    for word in vocabulary:
        if word in model:
            word_embedding = model[word]  # Get embedding for the word
            word_embeddings.append(word_embedding)
            valid_words.append(word)

    # Stack the embeddings into a matrix
    embedding_matrix = np.vstack(word_embeddings)

    # Compute pairwise cosine similarities for the entire matrix
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Initialize a list to store filtered words
    filtered_words = []

    # Loop over each word and its similarity scores
    for i, word in enumerate(valid_words):
        # Get all words that have similarity greater than the threshold
        similar_indices = np.where((similarity_matrix[i] > threshold) & (similarity_matrix[i] < 1.0))[0]
        if len(similar_indices) > 0:
            filtered_words.append(word)

    return filtered_words



# Step 4: Contextual Embeddings using Transformers
def load_transformer_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model


def get_embeddings(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


def compute_similarity(word1, word2, tokenizer, model):
    embedding1 = get_embeddings(word1, tokenizer, model)
    embedding2 = get_embeddings(word2, tokenizer, model)
    return np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# Step 5: Evaluation and Refinement
def evaluate_vocabulary(vocabulary, text_samples):
    print(f"Evaluating vocabulary of size {len(vocabulary)}")
    print(f"Vocabulary:\n{vocabulary}")
    coverage = sum(any(word in sample for sample in text_samples) for word in vocabulary)
    return coverage


# Main Execution
if __name__ == "__main__":
    # Initial Vocabulary Extraction
    most_common_words = extract_vocabulary(corpus=brown.words(), corpus_size=10000)
    print("Done! ------- most_common_words")

    # Semantic Coverage with WordNet
    basic_words = create_basic_words(most_common_words, vocabulary_size=voc_size)
    print("Done! ------- basic_words")

    # Filtering with Embeddings (GloVe)
    glove_file = 'glove.6B.100d.txt'  # Ensure this file is in the correct path 
    glove_model = load_glove_embeddings(glove_file)
    glove_filtered_words = filter_vocabulary(basic_words, glove_model)
    print("Done! ------- glove_filtered_words")

    filtered_basic_words = list(extract_vocabulary(glove_filtered_words, corpus_size=voc_size))
    print("Done! ------- filtered_basic_words")

    # Contextual Embeddings using Transformers
    tokenizer, transformer_model = load_transformer_model()


    df = pd.DataFrame({'vocabulary': filtered_basic_words})
    df.to_csv('vocab'+str(voc_size)+'.csv')
    csreparser.extract_words_in_format('vocab'+str(voc_size)+'.csv','vocab_words_formatted.txt')
    runpy.run_path('get_vocab_dict.py')
