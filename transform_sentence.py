import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch import sub
import UtilityFunctions as uf # Self written file

model, tokenizer = uf.get_bert_model()
vocab_embeddings = uf.get_pkl_vocab_embeddings()

# Returns the closest word in the vocabulary based on cosine similarity
# Gets a word embedding and a dictionary of vocabulary embeddings
def find_closest_word(word):

    word_embedding = uf.get_word_embedding(word, model, tokenizer)
   
    # Reshape for cosine similarity → shape: (1 × D)
    word_vec = word_embedding.reshape(1, -1)

    # Reshape for cosine similarity → shape: (V × D)
    vocab_matrix = np.vstack(list(vocab_embeddings.values()))

    sims = cosine_similarity(word_vec, vocab_matrix)[0]

    closest_idx = np.argmax(sims)

    closest_word = list(vocab_embeddings.keys())[closest_idx]
    return closest_word


# Reconstruct the sentence using substitutions
def reconstruct_sentence(sentence):
    
    # Words out of our limited vocabulary
    words_to_replace = uf.get_words_to_replace(sentence, uf.get_vocab())
    
    # Remove stop words requires a list of words
    words_to_replace = uf.remove_stop_words(words_to_replace)

    subs = {}
    for word in words_to_replace:
        closest_word = find_closest_word(word)
        if closest_word:
            subs[word] = closest_word
    
    transformed_sentence = []
    for word in words_to_replace:
        transformed_sentence.append(subs[word]) 

    return " ".join(transformed_sentence)

# Example usage
if __name__ == "__main__":
   
    results = {}
    sentences = []

    print("Processing sentences from SimilarityTests/TestSentences.txt ...")
    
    # read test sentences
    with open("SimilarityTests/TestSentences.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentence = line.split(". ", 1)[1] if ". " in line else line
                sentences.append(sentence)

    # process sentences
    for sentence in sentences:

        transformed = reconstruct_sentence(sentence)
        similarities = uf.similarity_checker(sentence, transformed)

        results[sentence] = {
            "transformed": transformed,
            "similarities": similarities
        }

    # write results
    uf.write_test_results(results)
    
    print("Processing complete. Results written to SimilarityTests/TestResults.txt")