from transformers import BertTokenizer, BertModel
import torch
from collections import Counter
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from sentence_similarity import compare_sentences
import pickle as pkl
from nltk.corpus import  stopwords , wordnet as wn
from gensim.models import KeyedVectors


# Returns the embedding vector for the given word
# Used in create_sentence.py , not to be confused with the  get_embedding function 
# Don't know the difference so we have both for now
# Originally used with Bert Model to get WORD embeddings of original sentence
def get_word_embedding(word, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    emb = outputs[1]  # The last hidden-state is the first element of the output tuple

    emb = emb.detach().numpy()
    emb = normalize(emb)

    return emb

# Returns a dictionary of word embeddings for all words in the sentence
def get_sentence_embeddings(sentence):
    words = set(sentence.split())  
    sentence_embeddings = {}
    for word in words:
        emb = get_word_embedding(word)
        if emb is not None:  # If the word has an embedding
            sentence_embeddings[word] = emb
    return sentence_embeddings

# Originally used in create_sentence.py to get VOCAB embeddings
def get_pkl_vocab_embeddings():
    with open("vocab_embeddings_dict.pkl", 'rb') as file:
        vocab_embeddings = pkl.load(file)
        
    return vocab_embeddings

# Get vocabulary from file after running vocab_semantic_filtering.py
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

def remove_stop_words(vocab):
    # filter stopwords (i.e. 'i', 'me', 'am', 'are', 'because', etc...)
    stop_words = set(stopwords.words('english'))
    # convert to lower case latter, filter non-alphabetic characters and remove stop words
    vocab = [word.lower() for word in vocab if word.isalpha() and word not in stop_words]
    
    return ' '.join(vocab)

# Returns the vocabulary ordered by commonality
def order_vocabulary(vocab):
     # count the requency of each word and select the most common words
    word_count = Counter(vocab)
    order_by_commonality_vocab = [word for word, _ in word_count.most_common(len(vocab))]

    return order_by_commonality_vocab

def get_hypernym(word):
    # find "hypernym" (more general term) for each word
    synsets = wn.synsets(word)
    hypernyms = set()
    for synset in synsets:
        hypernyms.update(lemma.name() for hypernym in synset.hypernyms() for lemma in hypernym.lemmas())
    return list(hypernyms)

def get_all_hypernyms(vocab):
    hypernyms = set()
    for word in vocab:
        hypernyms.update(get_hypernym(word))
    return list(hypernyms)  


def get_glove_vocab_embeddings(vocabulary):
    glove_file = 'glove.6B.100d.txt'
    model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    try:
        # Build a dictionary of word -> embedding for words present in the model
        vocab_embeddigs = {word: model[word] for word in vocabulary if word in model}
        if not vocab_embeddigs:
            raise ValueError("no words from vocabulary found in model")
    except Exception:
        print("no words found from vocabulary found in model")
        return {}
    return vocab_embeddigs

def get_bert_model():
    tokenizer = BertTokenizer.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    return model, tokenizer