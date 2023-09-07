import os
import cv2
import pandas as pd
import numpy as np
import shutil
import zipfile
import argparse


import warnings

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

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


def zipdir(path_train, path_model):
    ziph = zipfile.ZipFile('model.zip', 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk(path_train):
        for file in files:
            ziph.write(os.path.join(root, file))
    for root, dirs, files in os.walk(path_model):
        for file in files:
            ziph.write(os.path.join(root, file))
    ziph.close()

    os.system("sha1sum model.zip > signature.txt")


def get_data(train_path, validation_path):
    if validation_path:
        print("For training: ")
        data_train = image_dataset_from_directory(
            train_path,
            seed=42,
            image_size=(128, 128),
        )
        print("For validation: ")
        data_val = image_dataset_from_directory(
            validation_path,
            seed=42,
            image_size=(128, 128),
        )

        if data_train.class_names != data_val.class_names:
            raise Exception("Class names are not the same between"
                            "train and validation data")

        return (data_train, data_val)
    else:
        return image_dataset_from_directory(
            train_path,
            validation_split=0.2,
            subset="both",
            seed=42,
            image_size=(128, 128),
        )


def main_training(train_path, validation_path=None):
    print("Opening dataset...")
    data = get_data(train_path, validation_path)
    print()

    print("Generating model...")
    model = generate_model(data[0])
    print()

    print("Training model...")
    model.fit(data[0], epochs=8, validation_data=data[1])

    test_loss, test_acc = model.evaluate(data[1], verbose=2)
    print("test_loss : ", test_loss)
    print("test_acc : ", test_acc)
    model.save("model/model.h5")
    class_names = data[0].class_names
    df = pd.DataFrame(columns=class_names)
    df.to_csv("model/class_names.csv", index=False)

    zipdir("training_data", "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "batch_train", help="Path to the folder containing the training data"
    )
    parser.add_argument(
        "--batch_val",
        help="Path to the folder containing the validation data",
        default=None,
    )
    args = parser.parse_args()

    if not os.path.isdir(args.batch_train):
        raise Exception("Provided path doesn't exist or is not a folder")
    if args.batch_val and not os.path.isdir(args.batch_val):
        raise Exception("Provided path doesn't exist or is not a folder")
    main_training(args.batch_train, args.batch_val)
