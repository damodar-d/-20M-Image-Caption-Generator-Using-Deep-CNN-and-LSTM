import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_sequences(tokenizer, max_length, description_list, feature, vocab_size):
    input_img, input_seq, output_word = [], [], []
    for description in description_list:
        seq = tokenizer.texts_to_sequences([description])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            input_img.append(feature)
            input_seq.append(in_seq)
            output_word.append(out_seq)
    return np.array(input_img), np.array(input_seq), np.array(output_word)


def data_generator(descriptions, features, tokenizer, max_length, vocab_size):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                feature = features[key][0]
                input_img, input_seq, output_word = create_sequences(tokenizer, max_length, description_list, feature, vocab_size)
                for i in range(len(input_img)):
                    yield {'input_1': input_img[i], 'input_2': input_seq[i]}, output_word[i] 
    output_signature = (
            {
                'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32), 
                'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)  
        )    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.batch(64)