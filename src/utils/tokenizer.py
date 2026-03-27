from tensorflow.keras.preprocessing.text import Tokenizer
from helpers import dic_to_list

def create_tokenizer(descriptions):
    desc_list = dic_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

