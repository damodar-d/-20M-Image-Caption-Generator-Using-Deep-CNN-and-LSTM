from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.applications import Xception

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

from pickle import load
import numpy as np

from PIL import Image 

import argparse
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout, Add
from tensorflow.keras.models import Model

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i','--image', required = True, help = 'Image')
args = vars(arg_parser.parse_args())
img_path = args['image']
img_path = "./" + img_path

import tensorflow
print("Tensorflow version: ", tensorflow.__version__)


def extract_features(filename, model):
  
    image = Image.open(filename)
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4 :
        image = image[... , 3]
    image = np.expand_dims(image, axis = 0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_description(model, tokenizer, image, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen = max_length)
        prediction = model.predict([image, sequence], verbose = 0)
        prediction = np.argmax(prediction)
        word = word_for_id(prediction, tokenizer)
        if word is None:
            break
        in_text = in_text +" " + word
        if word == 'end':
            break
    return in_text

def define_model(vocab_size, max_length):

    #CNN model from which we will extract the features from the image
    # from 2048 nodes to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    #LSTM model which will take the caption as input and output a vector of 256 nodes
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model which will take the output from both the CNN and LSTM and output
    # a probability distribution over the vocabulary

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

max_length = 32
tokenizer = load(open('generated\serialized\\tokenizer\\tokenizer.p',"rb"))
vocab_size = len(tokenizer.word_index) + 1

model = define_model(vocab_size = vocab_size, max_length = max_length)
model.load_weights('models\checkpoints\epoch_19.h5')

xception_model = Xception(include_top = False, pooling = "avg")

photo = extract_features(img_path, xception_model)

img = Image.open(img_path)

description = generate_description(model, tokenizer,photo, max_length)

print(description)