import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import UtilityFunctions as uf # Self written file

# Returns the closest word in the vocabulary based on cosine similarity
# Gets a word embedding and a dictionary of vocabulary embeddings
def find_closest_word(word):
    model, tokenizer = uf.get_bert_model() 
    word_embedding = uf.get_word_embedding(word, model, tokenizer)
    
    vocab_embeddings = uf.get_pkl_vocab_embeddings() 
   
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
    
    # Remove stop words requires a list of words
    fixed_Sentence = uf.remove_stop_words(sentence.split())
    
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