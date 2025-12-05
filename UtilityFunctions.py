from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from sentence_similarity import compare_sentences
import pickle as pkl

tokenizer = BertTokenizer.from_pretrained('setu4993/LaBSE')
model = BertModel.from_pretrained('setu4993/LaBSE')

# Returns the embedding vector for the given word
# Used in create_sentence.py , not to be confused with the  get_embedding function 
# Don't know the difference so we have both for now
def get_word_embedding(word):
    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    english_embeddings = outputs[1]  # The last hidden-state is the first element of the output tuple

    english_embeddings = english_embeddings.detach().numpy()
    english_embeddings = normalize(english_embeddings)

    return english_embeddings

# Returns a dictionary of word embeddings for all words in the sentence
def get_sentence_embeddings(sentence):
    words = set(sentence.split())  
    sentence_embeddings = {}
    for word in words:
        emb = get_word_embedding(word)
        if emb is not None:  # If the word has an embedding
            sentence_embeddings[word] = emb
    return sentence_embeddings


def get_vocab_embeddings():
    with open("vocab_embeddings_dict.pkl", 'rb') as file:
        vocab_embeddings = pkl.load(file)
        
    return vocab_embeddings

def get_vocab():
    with open('vocab_words_formatted.txt', 'r') as file:
      return eval(file.read())
    

def load_transformer_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

# This appeared in vocabulary_semantic_filtering.py Step 4.
# Used instead of get_word_embedding
# Don't know the difference yet 
# This is NOT used in create_sentence.py
#def get_embedding(word):
    tokenizer, model = load_transformer_model()
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Compares the original and transformed sentences 
def similarity_checker(original_sentence, transformed_sentence):
    similarities = compare_sentences(original_sentence, transformed_sentence)
    print(f"Cosine Similarity (Sentence-BERT): {similarities['cosine_similarity_sentences_BERT']}")
    print(f"jaccard Similarity (Sentence-BERT): {similarities['jaccard_similarity_BERT']}")
