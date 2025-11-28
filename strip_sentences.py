import pandas as pd

data = pd.read_csv('SICK.txt', sep='\t')
pd.set_option('display.max_columns', None)
sub_data = data[['pair_ID', 'sentence_A', 'sentence_B', 'relatedness_score']]

strip_sentences = {}
for index, row in sub_data.iterrows():
    sentenceA = row['sentence_A']
    sentenceB = row['sentence_B']
    if sentenceA not in strip_sentences:
        strip_sentences[sentenceA] = sentenceA
    if sentenceB not in strip_sentences:
        strip_sentences[sentenceB] = sentenceB

print(len(strip_sentences))
df = pd.DataFrame(strip_sentences.keys(), columns=['sentences'])

df.to_csv('SICK_sentences.csv')
