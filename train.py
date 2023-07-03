import os
import sys
import cv2
import pandas as pd
import numpy as np
from Distribution import load_images_from_directory
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

            # Augment image using Augmentation.py
            # Transform image using only two methods from Transformation.py (blur & mask)
            # -> add result of augmentation and transformation in lists images and images_paths
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
            input_shape=(256, 256, 3),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(len(dataset.class_names), activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def main_training(path):
    data = image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(256, 256),
    )

    model = generate_model(data[0])

    model.fit(data[0], epochs=6, validation_data=data[1])

    test_loss, test_acc = model.evaluate(data[1], verbose=2)
    print("test_loss : ", test_loss)
    print("test_acc : ", test_acc)
    model.save("model/model.h5")
    class_names = data[0].class_names
    df = pd.DataFrame(columns=class_names)
    df.to_csv("model/class_names.csv", index=False)

    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print("loss : ", test_loss)
    # print("acc : ", test_acc)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Missing path argument")
    path = sys.argv[1]
    if not os.path.isdir(path):
        raise Exception("Provided path doesn't exist or is not a folder")
    main_training(path)
