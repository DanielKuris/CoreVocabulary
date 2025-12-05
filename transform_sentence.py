import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import UtilityFunctions as uf # Self written file

# Gets a sentence and a vocabulary set
# Returns the sentence without stop words (unless they are in the vocabulary!!!)
def get_filtered_sentence(sentence, vocab):
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    filtered_sentence = [
        w for w in words if w.lower() not in stop_words or w in vocab # Keep if in vocab
    ]
    return ' '.join(filtered_sentence)



# Returns the closest word in the vocabulary based on cosine similarity
# Gets a word embedding and a dictionary of vocabulary embeddings
def find_closest_word(word):
    
    word_embedding = uf.get_word_embedding(word)
    
    vocab_embeddings = uf.get_vocab_embeddings() # Consider cachinng in future!!! Can't now because where tf is create sentence called from?
   
    # Reshape for cosine similarity → shape: (1 × D)
    word_vec = word_embedding.reshape(1, -1)

    # Reshape for cosine similarity → shape: (V × D)
    vocab_matrix = np.vstack(list(vocab_embeddings.values()))

    sims = cosine_similarity(word_vec, vocab_matrix)[0]

    best_idx = np.argmax(sims)

    vocab_words = list(vocab_embeddings.keys())
    
    return vocab_words[best_idx]


# Reconstruct the sentence using substitutions
# Gets a sentence, vocabulary, and dictionaries of embeddings
def reconstruct_sentence(sentence):
    vocab = uf.get_vocab() # Consider cachinng in future!!! Can't now because where tf is create sentence called from?
    
    fixed_Sentence = get_filtered_sentence(sentence, vocab)
    
    subs = {}
    for word in fixed_Sentence:
        
        closest_word = find_closest_word(word)
        if closest_word:
            subs[word] = closest_word
    
    transformed_sentence = [
        subs.get(word, word) for word in sentence
    ]

    return " ".join(transformed_sentence)

# Example usage
if __name__ == "__main__":
    sentence = sys.argv[1]
    transformed_sentence = reconstruct_sentence(sentence)

    print("Original Sentence:", sentence)
    print("Transformed Sentence:", transformed_sentence)
    uf.similarity_checker(sentence, transformed_sentence)