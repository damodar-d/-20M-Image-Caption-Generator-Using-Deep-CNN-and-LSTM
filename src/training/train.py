from ..utils.constants import weights_saving_path
from ..utils.generator import data_generator
from ..utils.constants import dataset_text_dir_name, token_filename
from ..utils.loader import all_img_captions
from model import define_model
from ..utils.loader import load_clean_descriptions
from ..utils.feature_engineering import load_features
from ..utils.loader import load_photos
from ..model import define_model
from ..utils.tokenizer import create_tokenizer
from pickle import dump


max_length = 32
train_images = load_photos(token_filename)
token_filename = dataset_text_dir_name + '/Flickr8k.token.txt'

# 1. Generate Descriptions
descriptions = all_img_captions(token_filename)
print("Length of descriptions: ", len(descriptions))

#2. Load Clean Descriptions from file.
train_descriptions = load_clean_descriptions("descriptions.txt", train_images)
print("Length of training descriptions: ", len(train_descriptions))
train_features = load_features(train_images)


# 3. Create Tokenizer
tokenizer  = create_tokenizer(train_descriptions)
dump(tokenizer, open("tokenizer.p","wb"))
vocab_size = len(tokenizer.word_index) + 1


#Create Model
model = define_model(vocab_size, max_length)
model.summary()

epochs = 20
steps_per_epoch = 5
# os.mkdir("models")
for i in range(epochs):
    dataset = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    model.fit(dataset, epochs=10, steps_per_epoch=steps_per_epoch, verbose=1)
    model.save("models/model_" + str(i) + ".h5")