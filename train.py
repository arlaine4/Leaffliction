import os
import sys
import cv2
import pandas as pd
import numpy as np
from Distribution import load_images_from_directory
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from tensorflow.keras import layers, models

import warnings
warnings.filterwarnings('ignore')

TARGETS_DICT = {}

def fill_target_dict(dir_and_images):
    global TARGETS_DICT
    for key, val in dir_and_images.items():
        key = key.split('/')[-1]
        TARGETS_DICT[key] = len(TARGETS_DICT.keys())


def prepare_dataset(dir_and_images):
    fill_target_dict(dir_and_images)
    train_df = pd.DataFrame(columns=['target', 'image_path', 'image_data'])
    for key, val in dir_and_images.items():
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
    return train_df


def generate_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def load_image_from_path(image_path):
    image = cv2.imread(image_path)
    return image


def main_training(path):
    final_dirs = {}
    for root, dirs, files in os.walk(path):
        if not dirs:
            final_dirs[root] = load_images_from_directory(root)
    train_df = prepare_dataset(final_dirs)
    train_df.to_csv('test.csv')
    model = generate_model()
    
    X_train, X_test, y_train, y_test = train_test_split(np.array(train_df['image_data'], dtype=np.uint8),
                                                        np.array(train_df['target'], dtype=np.uint8),
                                                        test_size=.2)
    print(model.summary())
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("loss : ", test_loss)
    print("acc : ", test_acc)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Missing path argument")
    path = sys.argv[1]
    if not os.path.isdir(path):
        raise Exception("Provided path doesn't exist or is not a folder")
    main_training(path)
