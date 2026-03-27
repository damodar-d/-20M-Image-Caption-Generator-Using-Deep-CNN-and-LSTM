import os
from constants import dataset_images_dir_name

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split('\n')[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images_dir_name, photo))]
    return photos_present


def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        image, caption = caption.split('\t')
        if image[:-2] not in descriptions:
            descriptions[image[:-2]] = [caption]
        else:
            descriptions[image[:-2]].append(caption)
    return descriptions

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split('\n'):
        words = line.split()
        if len(words) < 1:
            continue
        image,image_caption = words[0], words[1:]
        if image in photos:
            descriptions[image] = []
            desc = '<start> '+" ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions