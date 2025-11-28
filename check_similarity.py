import pandas as pd
from sentence_similarity import compare_sentences
from create_sentence import create_sentence
import time

curr = "1000"
embbedings_path = "vocabularies/pkls_embbedings/vocab" + curr + "embeddings.pkl"
output_path = "vocabularies/results/" + curr

data = pd.read_csv('SICK_sentences.csv')

sub_data = data[['sentences']]
OG_sentences = list()
gen_sentences = list()
cosine_BERT = list()
jaccard_BERT = list()

for index, row in sub_data.iterrows():
    print(f'Index: {index}')
    OG_sentence = row['sentences']
    gen_sentence = create_sentence(OG_sentence, embbedings_path)
    OG_sentences.append(OG_sentence)
    gen_sentences.append(gen_sentence)

    # original pair
    similarities = compare_sentences(OG_sentence, gen_sentence)
    bert = similarities['cosine_similarity_sentences_BERT']
    jacard = similarities['jaccard_similarity_BERT']
    cosine_BERT.append(bert)
    jaccard_BERT.append(jacard)

    print(f'OG_sentence: {OG_sentence}')
    print(f'gen_sentence: {gen_sentence}')
    print(f'BERT_score: {bert}')
    print(f'Jaccard_score: {jacard}')
    print()
new_data = pd.DataFrame()
new_data['OG_sentences'] = OG_sentences
new_data['gen_sentences'] = gen_sentences
new_data['cosine_similarity_BERT'] = cosine_BERT
new_data['jaccard_similarity_BERT'] = jaccard_BERT

current_time = time.strftime("%Y%m%d-%H%M%S")
avg_cosine = sum(new_data['cosine_similarity_BERT']) / len(new_data)
avg_jaccard = sum(new_data['jaccard_similarity_BERT']) / len(new_data)
print(f'\n\nAverage cosine score: {avg_cosine}')
print(f'\n\nAverage Jaccard score: {avg_jaccard}')
new_data.to_csv(f'{output_path}similarities_cosineAVG{avg_cosine:.4f}_jaccardAVG{avg_jaccard:.4f}_{current_time}.csv',
                index=False)

# print(sub_data.head())
