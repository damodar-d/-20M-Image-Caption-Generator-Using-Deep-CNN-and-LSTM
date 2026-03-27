
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

print("Base Directory: ",BASE_DIR)

dataset_text_dir_name = Path.joinpath(BASE_DIR , "data/processed/training")
dataset_images_dir_name = Path.joinpath(BASE_DIR , "data/raw")

gen_cleaned_descriptions_path = Path.joinpath(BASE_DIR , 'generated/descriptions')

# token_filename = dataset_text_dir_name + '/Flickr_8k.trainImages.txt'

pre_trained_weights_saving_path = Path.joinpath(BASE_DIR , "pre-trained/xception/xception.h5")

final_weights_saving_path = Path.joinpath(BASE_DIR , 'models/final')

print(pre_trained_weights_saving_path)
