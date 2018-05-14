
from keras.applications import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from imutils import paths
import h5py
import os
import numpy as np
from tqdm import tqdm

def process_images(image_files=[], input_shape=(350, 350, 3)):
    images = []
    for image_file in image_files:        
        img = preprocess_input(img_to_array(load_img(image_file)))
        images.append(img)
    return np.stack(images)        


def extract_feature(image_files=[], input_shape=(350, 350, 3)):
    images = []
    for image_file in image_files:        
        img = preprocess_input(img_to_array(load_img(image_file)))
        images.append(img)

    images = np.stack(images)        
    model = ResNet50(weights="imagenet", 
                     include_top=False, 
                     input_shape=input_shape,
                     pooling="avg")
    return model.predict(images)

def process_FBP5500(FBP5500_path, db_file, input_shape=(350, 350, 3), batch_size=128):
    image_path = FBP5500_path+"/Images"
    feature_file = FBP5500_path+"/train_test_files/All_labels.txt"

    #1. Process feature dictionary
    feature_dict = {}
    with open(feature_file) as f:
        for line in f:
            (key, val) = line.split()
            feature_dict[key] = float(val)

    #2. Process Image files
    all_images = list(paths.list_images(image_path))
    db_file = h5py.File(db_file, "w")


    batch_labels = []
    batch_images = []
    
    model = ResNet50(weights="imagenet", 
                     include_top=False, 
                     input_shape=input_shape,
                     pooling="avg")

    image_data = db_file.create_dataset("features", shape=(len(all_images), model.layers[-1].output.shape[-1]))
    label_data = db_file.create_dataset("labels", shape=(len(all_images),))
    
    i = 0
    for image_file in tqdm(all_images):
        label_key = image_file.split(os.path.sep)[-1]
        img = preprocess_input(img_to_array(load_img(image_file)))
        
        batch_images.append(img)
        batch_labels.append(feature_dict[label_key])

        if len(batch_images) == batch_size:
            batch_images = np.stack(batch_images)
            batch_labels = np.stack(batch_labels)
            
            # print(batch_images.shape)
            # print(batch_labels.shape)

            features = model.predict(batch_images)

            image_data[i:i+batch_size] = features
            label_data[i:i+batch_size] = batch_labels

            batch_images = []
            batch_labels = []
            i += i+batch_size

    # Remaining items 
    if len(batch_images) > 0:
        print("[INFO]Processing {} remaining items".format(len(batch_images)))
        batch_images = np.stack(batch_images)
        batch_labels = np.stack(batch_labels)
        
        features = model.predict(batch_images)

        image_data[i:i+len(batch_images)] = features
        label_data[i:i+len(batch_labels)] = batch_labels
        
    db_file.close()