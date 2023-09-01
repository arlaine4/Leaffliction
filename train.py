import os
import sys
import cv2
import pandas as pd
import numpy as np
import shutil


import warnings

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

from Augmentation import main_augmentation
from Transformation import batch_transform

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
                    TARGETS_DICT[key_target_name]
                    for i in range(len(images_paths))
                ],
                "image_path": images_paths,
                "image_data": images_data,
            }
        )
    train_df = pd.concat([train_df, df])
    train_df.to_csv("test.csv")
    return train_df


def generate_model(dataset):
    model = models.Sequential()
    model.add(layers.Rescaling(1.0 / 255))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(32, (1, 1), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
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
    for key in list(distrib.keys()):
        if distrib[key] > mean:
            not_distrib[key] = distrib[key]
            del distrib[key]
    return distrib, not_distrib


def create_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)


def equalizes_dataset(path, q=0.75):
    # compute the number of images of each dir
    dir_and_images = {}
    for root, dirs, files in os.walk(path):
        if not dirs:
            dir_and_images[root] = len(files)

    # compute the quantile
    quantile = np.quantile(list(dir_and_images.values()), q)

    create_dir("training_data")
    for root, dirs, files in os.walk(path):
        if not dirs:
            if len(files) < quantile:
                create_dir(os.path.join("training_data", root.split("/")[-1]))
                for file in files:
                    shutil.copy(
                        os.path.join(root, file),
                        os.path.join("training_data", root.split("/")[-1]),
                    )

                main_augmentation(root, "batch", training=True)

                # complete with augmented_directory
                missing = int(quantile) - len(files)
                print("Missing : {} for {}".format(missing, root))

                for c_root, c_dirs, c_files in os.walk(
                    "augmented_directory" + "/" + root.split("/")[-1]
                ):
                    if not c_dirs:
                        c_files = os.listdir(c_root)
                        c_files = [f for f in c_files if "original" not in f]
                        for file in np.random.choice(c_files, missing,
                                                     replace=False):
                            shutil.copy(
                                os.path.join(c_root, file),
                                os.path.join("training_data",
                                             c_root.split("/")[-1]),
                            )
            else:
                create_dir(os.path.join("training_data", root.split("/")[-1]))
                for file in np.random.choice(files, int(quantile),
                                             replace=False):
                    shutil.copy(
                        os.path.join(root, file),
                        os.path.join("training_data", root.split("/")[-1]),
                    )

            path = os.path.join("training_data", root.split("/")[-1])
            batch_transform(path, "training_data", training=True)


def main_training(path):
    equalizes_dataset(path)

    data = image_dataset_from_directory(
        "training_data",
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
