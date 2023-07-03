import os
import sys
import cv2
import pandas as pd
import numpy as np
from Distribution import load_images_from_directory
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

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
        key_target_name = target.split("/")[-1]
        images_paths = []
        images_data = []
        for path in paths:
            image_path = os.path.join(target, path)
            images_paths.append(image_path)

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
    train_df.to_csv("test.csv")
    return train_df
    # image_path = os.path.join()
    # image_path = os.path.join(path
    """for key, val in dir_and_images.items():
        # Extracting target class from path
        key_target = key.split('/')[-1]
        images_paths = []
        values = []
        for value in val:
            # Creating valid path to image
            image_path = os.path.join(key, value)
            # Updating list of images paths
            images_paths.append(image_path)
            # Updating list of images metadata
            values.append(load_image_from_path(image_path))

        # Creating dataframe for each target
        df = pd.DataFrame({'target': [TARGETS_DICT[key_target] for i in range(len(val))],
                           'image_path': images_paths,
                           'image_data': values})
        # Adding new target dataframe to train_df
        train_df = pd.concat([train_df, df])
    print(train_df.head(10))
    return train_df"""


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


def load_image_from_path(image_path):
    image = cv2.imread(image_path)
    return image


def main_training(path):
    dataset = image_dataset_from_directory(path)
    model = generate_model(dataset)

    model.build(input_shape=(256, 256, 3))
    print(model.summary())
    model.fit(dataset, epochs=10)
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
