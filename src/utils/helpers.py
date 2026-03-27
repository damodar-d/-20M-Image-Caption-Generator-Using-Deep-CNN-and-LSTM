import time 
from tensorflow.keras.utils import get_file 
import string

def download_with_retry(url, filename, max_retries = 3):
    for attempt in range(max_retries):
        try:
            return get_file(filename, url)
        except Exception as e:
            if attempt == max_retries - 1 :
                raise e
            print("Download attempt failed")
            time.sleep(3)

def save_descriptions(descriptions,filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key+'\t'+desc)
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close


def generate_inMemory_vocabulary(descriptions):
    vocab = set()
    for key in descriptions:
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

def clean_text(captions):
    table = str.maketrans('','',string.punctuation)
    for image, caps in captions.items():
        for i, image_caption in enumerate(caps):
            image_caption.replace("-","")
            desc = image_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]

            image_caption = ' '.join(desc)
            captions[image][i] = image_caption
    return captions

def dic_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def max_length_descriptions(descriptions):
    desc_list = dic_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)


import requests
from constants import pre_trained_weights_saving_path
url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5"
def fetch_pretrained_xception_weights():
    response = requests.get(url, stream = True)

    with open(pre_trained_weights_saving_path,'wb') as f:
        for chunk in response.iter_content(chunk_size = 8192):
            if chunk: 
                f.write(chunk)

    print("Download Complete")

fetch_pretrained_xception_weights()