from tensorflow.keras.layers import Dropout, Input, Dense, LSTM, Embedding
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model


def define_model(vocab_size, max_length):

    #CNN model from which we will extract the features from the image
    # from 2048 nodes to 256 nodes
    inputs1 = Input(shape=(2048,), name = 'input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    #LSTM model which will take the caption as input and output a vector of 256 nodes
    inputs2 = Input(shape=(max_length,), name = 'input_2')
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