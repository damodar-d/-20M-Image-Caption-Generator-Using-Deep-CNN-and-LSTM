from helpers import download_with_retry


dataset_text_dir_name = "data/processed/training"
dataset_images_dir_name = "data/raw"

gen_cleaned_descriptions_path = 'generated/descriptions' 

token_filename = dataset_text_dir_name + '/Flickr_8k.trainImages.txt'

weights_url = "file:///Users/abhiramdeshpande/Desktop/development/ML Projects/Image Caption Generator/xception_weights.h5"
weights_path = download_with_retry(weights_url, "xception_weights.h5")


weights_saving_path = 'models/final'
print(weights_path)