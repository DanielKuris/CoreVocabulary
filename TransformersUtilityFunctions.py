from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import normalize

tokenizer = BertTokenizer.from_pretrained('setu4993/LaBSE')
model = BertModel.from_pretrained('setu4993/LaBSE')


def get_vec(text_input):
    input_ids = torch.tensor(tokenizer.encode(text_input)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    english_embeddings = outputs[1]  # The last hidden-state is the first element of the output tuple

    english_embeddings = english_embeddings.detach().numpy()
    english_embeddings = normalize(english_embeddings)

    return english_embeddings
