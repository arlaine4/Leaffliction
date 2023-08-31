import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys

from tensorflow.keras.models import load_model
from Transformation import Transformation, options


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default=False,
                        help='Make prediction from an image path')
    parser.add_argument('-b', '--batch', default=False,
                        help='Make prediction from a directory path')
    parser.add_argument('-m', '--model', default='model/model.h5',
                        help='Model path')
    parser.add_argument('-cn', '--class_names', default='model/class_names.csv',
                        help='class_names.csv path')
    options = parser.parse_args()
    return options


def verif_args(args):
    if not args.image and not args.batch:
        raise Exception("Please provide -b or -m to select prediction source")
    if args.image and args.batch:
        raise Exception('You must choose between batch or image mode, not both')
    if args.image:
        if not os.path.exists(args.image):
            raise Exception("Image path is invalid")
        if not args.image.endswith('.JPG'):
            raise Exception("Invalid file extension")
        return args.image, 'image'
    if args.batch:
        if not os.path.isdir(args.batch):
            raise Exception('Batch folder path is invalid')
        if len(os.listdir(args.batch)) == 0:
            raise Exception('Empty folder, please provide a path containing files in it')
        return args.batch, 'batch'


def make_predictions(pred_path, mode, model, class_names):
    if mode == "image":
        image = cv2.imread(pred_path)
        image_resize = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

        y_hat = model.predict(np.expand_dims(image_resize, axis=0))
        class_name_prediction = class_names[np.argmax(y_hat)]
        options_img = options(pred_path, debug=None)
        tr = Transformation(options_img)
        masked = tr.masked()

        plt.suptitle(f"Prediction: {class_name_prediction}")

        plt.subplot(1, 2, 1)
        plt.imshow(image_resize)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(masked)
        plt.axis("off")
        plt.show()
    else:
        # Batch mode used on images/<class>/<sub_class_name>
        images = os.listdir(pred_path)
        y_hat = []
        valid_classes = []
        correct_preds = 0
        for image in images:
            image_loaded = cv2.imread(os.path.join(pred_path, image))
            image_resize = cv2.resize(image_loaded, (128, 128), interpolation=cv2.INTER_AREA)

            y_hat = model.predict(np.expand_dims(image_resize, axis=0))
            class_name_prediction = class_names[np.argmax(y_hat)]
            true_class = pred_path.split('/')[-1]
            if class_name_prediction == true_class:
                correct_preds += 1
                print(f"Prediction for {image} : \033[32m{class_name_prediction}\033[0m")
            else:
                print(f"Prediction for {image} : \033[31m{class_name_prediction}\033[0m")
        print(f"Got {int((correct_preds / len(images)) * 100)}% valid prediction out of {len(images)} images")




if __name__ == "__main__":
    args = parse_args()
    prediction_path, mode = verif_args(args)

    try:
        model = load_model(args.model)
    except FileNotFoundError:
        sys.exit("Model not found, please run train.py first")
    try:
        class_names = np.loadtxt(args.class_names, dtype=str, delimiter=',')
    except FileNotFoundError:
        sys.exit("Class names file not found, please run train.py first")
    make_predictions(prediction_path, mode, model, class_names)
