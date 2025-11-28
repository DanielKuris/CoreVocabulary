import pandas as pd
from sentence_similarity import compare_sentences
from nltk.corpus import stopwords

# Load NLTK stop words
stop_words = set(stopwords.words('english'))

data = pd.read_csv('SICK.txt', sep='\t')
pd.set_option('display.max_columns', None)
sub_data = data[['pair_ID', 'sentence_A', 'sentence_B', 'relatedness_score']]
# print(sub_data.head())
cosine_BERT = list()
jaccard_BERT = list()
B_no_stopwords_cosine_BERT = list()
B_no_stopwords_jaccard_BERT = list()

# normalize 'relatedness_score'
sub_data['relatedness_score'] = sub_data['relatedness_score'].astype(float)
sub_data['relatedness_score'] = (sub_data['relatedness_score'] - sub_data['relatedness_score'].min()) / (
        sub_data['relatedness_score'].max() - sub_data['relatedness_score'].min())

for index, row in sub_data.iterrows():
    print(f'Index: {index}')
    sentence_A = row['sentence_A']
    sentence_B = row['sentence_B']

    # original pair
    similarities = compare_sentences(sentence_A, sentence_B)
    bert = similarities['cosine_similarity_sentences_BERT']
    jacard = similarities['jaccard_similarity_BERT']
    cosine_BERT.append(bert)
    jaccard_BERT.append(jacard)

    # second sentence without stopwords
    sentence_B_no_stopwords = ' '.join([w for w in sentence_B.split() if w.lower() not in stop_words])
    # print(f'\n\n\n{sentence_B}\n{sentence_B_no_stopwords}\n\n\n')
    B_no_stopwords_similarities = compare_sentences(sentence_A, sentence_B_no_stopwords)
    B_no_stopwords_bert = B_no_stopwords_similarities['cosine_similarity_sentences_BERT']
    B_no_stopwords_jacard = B_no_stopwords_similarities['jaccard_similarity_BERT']
    B_no_stopwords_cosine_BERT.append(B_no_stopwords_bert)
    B_no_stopwords_jaccard_BERT.append(B_no_stopwords_jacard)

    print(f'relatedness_score: {row["relatedness_score"]}')
    print(f'BERT_score: {bert}')
    print(f'Jaccard_score: {jacard}')
    print(f'B_no_stopwords_cosine_BERT: {B_no_stopwords_bert}')
    print(f'B_no_stopwords_jaccard_BERT: {B_no_stopwords_jacard}')
    print()

sub_data['cosine_similarity_BERT'] = cosine_BERT
sub_data['jaccard_similarity_BERT'] = jaccard_BERT
sub_data['B_no_stopwords_cosine_BERT'] = B_no_stopwords_cosine_BERT
sub_data['B_no_stopwords_jaccard_BERT'] = B_no_stopwords_jaccard_BERT
sub_data.to_csv('new_sub_data.csv', index=False)

# print(sub_data.head())
