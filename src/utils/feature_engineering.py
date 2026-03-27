from tensorflow.keras.applications.xception import Xception
import os 
from PIL import Image
from tensorflow.keras.utils import to_categorical
from constants import weights_path
import numpy as np
from pickle import load

def extract_features(directory):
    model = Xception(include_top = False, pooling = 'avg', weights = weights_path)
    features = {}
    valid_images_extension = ['.jpg','.jpeg','.png']
    for img in os.listdir(directory):
        extension = os.path.splitext(img)[1].lower()
        if extension not in valid_images_extension:
            continue
        filename = directory + '/' + img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image, axis= 0)
        image = image/127.5
        image = image - 1.0

        feature = model.predict(image)
        print(feature)
        features[img] = feature
    print("Extracted features: ",features)
    return features

def load_features(train_images):
    all_features = load(open("features.p", 'rb'))
    print("All Features: ", all_features)
    features = {k: all_features[k] for k in train_images}
    return features