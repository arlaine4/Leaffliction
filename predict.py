import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
from random import shuffle

from tensorflow.keras.models import load_model
from Transformation import transform_tmp, delete_tmp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image", default=False,
        help="Make prediction from an image path"
    )
    parser.add_argument(
        "-b", "--batch", default=False,
        help="Make prediction from a directory path"
    )
    parser.add_argument("-m", "--model", default="model/model.h5",
                        help="Model path")
    parser.add_argument(
        "-cn",
        "--class_names",
        default="model/class_names.csv",
        help="class_names.csv path",
    )
    options = parser.parse_args()
    return options


def verif_args(args):
    if not args.image and not args.batch:
        raise Exception("Please provide -b or -m to select prediction source")
    if args.image and args.batch:
        raise Exception("Choose between batch or image mode, not both")
    if args.image:
        if not os.path.exists(args.image):
            raise Exception("Image path is invalid")
        if not args.image.endswith(".JPG"):
            raise Exception("Invalid file extension")
        return args.image, "image"
    if args.batch:
        if not os.path.isdir(args.batch):
            raise Exception("Batch folder path is invalid")
        if len(os.listdir(args.batch)) == 0:
            raise Exception(
                "Empty folder, please provide a path containing files in it"
            )
        return args.batch, "batch"


def predict_image(image, model):
    image = np.array(image)
    image_resize = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    return model.predict(np.expand_dims(image_resize, axis=0), verbose=0)


def plot_prediction(image, image_masked, class_name_prediction, img_path):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, figsize=(12, 8))
    ax0.imshow(image)
    ax0.set_title(f"Image : {img_path}", fontsize=10)
    ax1.imshow(image_masked)
    ax1.set_title("Masked image", fontsize=10)
    fig.suptitle(f"Class predicted : {class_name_prediction}")
    plt.show()


def make_prediction_on_image(image_path, model, class_names, plot=False):
    images_transform = transform_tmp(image_path)
    predictions = []

    for key, image in images_transform.items():
        predictions.append(predict_image(image, model))

    mean = np.mean(predictions, axis=0)
    class_name_prediction = class_names[np.argmax(mean)]

    if plot:
        plot_prediction(
            images_transform["original"],
            images_transform["masked"],
            class_name_prediction,
            image_path
        )

    return class_name_prediction


def make_prediction_on_batch(dir_path, model, class_names):
    images = os.listdir(dir_path)
    # Shuffling images from batch directory so we can run multiple
    # test on each batch and making predictions on different
    # images every time
    shuffle(images)
    predictions = 0
    valid = 0
    if dir_path[-1] == '/':
        dir_path = dir_path[:-1]
    true_class = dir_path.split("/")[-1]

    for i, image in enumerate(images):
        cp = make_prediction_on_image(
            os.path.join(dir_path, image), model, class_names, plot=False
        )
        predictions += 1
        if cp == true_class:
            valid += 1
            print(f"Prediction for {image} : \033[32m{cp}\033[0m")
        else:
            print(f"Prediction for {image} : \033[31m{cp}\033[0m")
        if i == 99:
            break

    print(f"Got {int((valid / predictions) * 100)}% valid prediction")
    print("{}/{} valid predictions".format(valid, predictions))


def main(path, mode, model, class_names):
    if mode == "image":
        make_prediction_on_image(path, model, class_names, plot=True)
    else:
        make_prediction_on_batch(path, model, class_names)
    delete_tmp()


if __name__ == "__main__":
    args = parse_args()
    prediction_path, mode = verif_args(args)

    try:
        model = load_model(args.model)
    except FileNotFoundError:
        sys.exit("Model not found, please run train.py first")
    try:
        class_names = np.loadtxt(args.class_names, dtype=str, delimiter=",")
    except FileNotFoundError:
        sys.exit("Class names file not found, please run train.py first")
    main(prediction_path, mode, model, class_names)
