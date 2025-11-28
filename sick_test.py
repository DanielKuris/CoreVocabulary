import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Load the SICK dataset
df = pd.read_csv('SICK.txt', delimiter='\t', on_bad_lines='skip')
original_sentences = df['sentence_A']
transformed_sentences = df['sentence_B']

# Define function to compute Jaccard similarity
def jaccard_similarity(sent1, sent2):
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

# Define function to compute Cosine similarity
def cosine_similarity_sentences(sent1, sent2):
    vectorizer = CountVectorizer().fit_transform([sent1, sent2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

# Calculate Jaccard and Cosine similarities for each sentence pair
jaccard_similarities = []
cosine_similarities = []

for sent1, sent2 in zip(original_sentences, transformed_sentences):
    jaccard_sim = jaccard_similarity(sent1, sent2)
    cosine_sim = cosine_similarity_sentences(sent1, sent2)
    jaccard_similarities.append(jaccard_sim)
    cosine_similarities.append(cosine_sim)

# Calculate average similarities
average_jaccard = np.mean(jaccard_similarities)
average_cosine = np.mean(cosine_similarities)

print(f'Average Jaccard Similarity: {average_jaccard:.4f}')
print(f'Average Cosine Similarity: {average_cosine:.4f}')
