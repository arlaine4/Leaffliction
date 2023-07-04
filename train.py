import os
import sys
import cv2
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Augmentation import main_augmentation
from Transformation import batch_transform

import warnings

warnings.filterwarnings("ignore")

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
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
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
    for root, dirs, files in os.walk(path):
        if not dirs:
            distrib[root] = len(os.listdir(root))
    distrib = dict(sorted(distrib.items(), key=lambda x: x[1]))
    to_augment = len(list(distrib.keys())) // 2
    final_distrib = {k: distrib[k] for k in list(distrib)[:to_augment]}
    return final_distrib


def main_training(path):
    """folders_to_augment = get_list_of_folders_to_augment(path)
    print(folders_to_augment)
    for folder_path in folders_to_augment:
        print(f"calling main_augmentation for {folder_path}")
        main_augmentation(folder_path, "batch")"""

    #batch_transform(path, "transformed_directory")

    # Add call to transformation
    data = image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(128, 128),
    )

    model = generate_model(data[0])

    model.fit(data[0], epochs=10, validation_data=data[1])

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
