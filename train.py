import os
import sys
import cv2
import pandas as pd
import numpy as np
import shutil


import warnings
warnings.filterwarnings("ignore")

import sklearn
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix

from Augmentation import main_augmentation
from Transformation import batch_transform

TARGETS_DICT = {}


def fill_target_dict(dir_and_images):
    global TARGETS_DICT
    for key, val in dir_and_images.items():
        key = key.split("/")[-1]
        TARGETS_DICT[key] = len(TARGETS_DICT.keys())


def prepare_dataset(dir_and_images):
    fill_target_dict(dir_and_images)
    train_df = pd.DataFrame(columns=["target", "image_path"])

    for target, paths in dir_and_images.items():
        print(f"Generate dataset for {target}")
        # Extracting target name from folder path
        key_target_name = target.split("/")[-1]
        # List of image paths
        images_paths = []
        # List of image matrix data
        images_data = []
        # Going through each image path from a specific folder 'target'
        for path in paths:
            # Getting full image path
            image_path = os.path.join(target, path)
            images_paths.append(image_path)

            # Loading and preprocessing image
            image = cv2.imread(image_path, cv2.COLOR_RGB2BGR)
            image = np.array(image)
            images_data.append(image)

        df = pd.DataFrame(
            {
                "target": [
                    TARGETS_DICT[key_target_name] for i in range(len(images_paths))
                ],
                "image_path": images_paths,
                "image_data": images_data,
            }
        )
    train_df = pd.concat([train_df, df])
    # Add check to know if we should generate train_df or not
    train_df.to_csv("test.csv")
    return train_df


def generate_model(dataset):
    model = models.Sequential()
    model.add(layers.Rescaling(1.0 / 255))
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(128, 128, 3),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(
        layers.Conv2D(
            32, (7, 7), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(len(dataset.class_names), activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def get_list_of_folders_to_augment(path):
    distrib = {}
    not_distrib = {}

    mean = 0
    for root, dirs, files in os.walk(path):
        if not dirs:
            distrib[root] = len(os.listdir(root))
            mean += distrib[root]
    mean /= len(list(distrib.keys()))
    print('Mean number of images over all directories is : ', mean)
    for key in list(distrib.keys()):
        if distrib[key] > mean:
            not_distrib[key] = distrib[key]
            del distrib[key]
    print('Final distrib after clean : ', distrib)
    print('Final not_distrib after clean : ', not_distrib)
    print()
    return distrib, not_distrib


def prepare_final_dataset_directory(base_path):
    """
    Function used to generate a single big folder
    containing the base image, transformed and augmented versions
    of each image for each class.
    This folder will be used to generate training and validation
    dataset.
    """
    try:
        os.makedirs('training_data')
    except FileExistsError:
        pass
    for root, dirs, files in os.walk(base_path):
        if len(dirs) != 0 and len(files) == 0:
            for _dir in dirs:
                try:
                    os.makedirs(os.path.join('training_data', _dir))
                except FileExistsError:
                    shutil.rmtree(os.path.join('training_data', _dir))
                    os.makedirs(os.path.join('training_data', _dir))

    to_walk = ['transformed_directory', 'augmented_directory']

    for folder in to_walk:
        for root, dirs, files in os.walk(folder):
            if len(files) != 0:
                files = os.listdir(root)
                for file in files:
                    shutil.copy(os.path.join(root, file), \
                        os.path.join('training_data', root.split('/')[-1]))

def prepare_for_not_augmented(folders_to_not_augment):
    for folder in folders_to_not_augment:
        for root, dirs, files in os.walk(folder):
            if len(files) != 0:
                files = os.listdir(root)
                for file in files:
                    shutil.copy(os.path.join(root, file), \
                        os.path.join('training_data', root.split('/')[-1]))


def main_training(path):
    folders_to_augment, folders_to_not_augment = get_list_of_folders_to_augment(path)
    for folder_path in folders_to_augment:
        print(f"Augmenting {folder_path}:")
        main_augmentation(folder_path, "batch", training=True)
        print(f"Transforming {folder_path}:")
        batch_transform(folder_path, "transformed_directory", training=True)
        print()

    # Add call to transformation
    # new_path = os.path.join('augmented_directory', path.split('/')[-1])
    prepare_final_dataset_directory(path)
    prepare_for_not_augmented(folders_to_not_augment)

    data = image_dataset_from_directory(
        'training_data',
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(128, 128),
    )

    model = generate_model(data[0])
    model.fit(data[0], epochs=8, validation_data=data[1])


    test_loss, test_acc = model.evaluate(data[1], verbose=2)
    print("test_loss : ", test_loss)
    print("test_acc : ", test_acc)
    model.save("model/model.h5")
    class_names = data[0].class_names
    df = pd.DataFrame(columns=class_names)
    df.to_csv("model/class_names.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Missing path argument")
    path = sys.argv[1]
    if not os.path.isdir(path):
        raise Exception("Provided path doesn't exist or is not a folder")
    main_training(path)
