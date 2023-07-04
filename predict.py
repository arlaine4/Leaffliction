import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
import os
from tqdm import tqdm

from Transformation import Transformation, options


def batch_predict(src, model, class_names):
    # read dir
    data = image_dataset_from_directory(
        src,
        batch_size=1,
        image_size=(128, 128),
    )

    if len(data.class_names) != len(class_names):
        raise Exception("Class names not same length")
    for c in data.class_names:
        if c not in class_names:
            raise Exception(f"Class {c} not in class_names")

    predictions = {c: 0 for c in class_names}
    len_class = {c: 0 for c in class_names}

    # for each ellement in each class predict
    for _, (images, labels) in tqdm(enumerate(data)):
        # predict
        prediction = model.predict(images, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]

        # get the real class
        real_class = class_names[labels[0]]

        len_class[real_class] += 1
        if real_class == predicted_class:
            predictions[real_class] += 1

    # display results
    for c, v in predictions.items():
        print(c)
        print("Score: {}/{}".format(v, len_class[c]))
        print("Accuracy: ", v / len_class[c])
        print()


def predict_file(model, class_names, image):
    # Resizing image to match the input shape the model is expecting
    image_resize = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

    # Predict the class of the image
    prediction = model.predict(np.expand_dims(image_resize, axis=0))

    class_name_prediction = class_names[np.argmax(prediction)]

    opt = options(image, debug=None)
    tr = Transformation(opt)
    masked = tr.masked()

    plt.suptitle(f"Prediction: {class_name_prediction}")

    plt.subplot(1, 2, 1)
    plt.imshow(image_resize)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(masked)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction of images")
    parser.add_argument("src", type=str, help="Path to the image or dir")
    parser.add_argument(
        "-m", "--model", type=str, help="Path to the model", default="model/model.h5"
    )
    parser.add_argument(
        "-cn",
        "--class_names",
        type=str,
        help="Path to the class names file",
        default="model/class_names.csv",
    )
    args = parser.parse_args()

    # check if src is a dir
    if os.path.isdir(args.src):
        batch_predict(
            args.src,
            load_model(args.model),
            np.loadtxt(args.class_names, dtype=str, delimiter=","),
        )
        exit(0)
    else:
        if not args.src.endswith(".JPG"):
            exit("Not a JPG file")
        predict_file(
            load_model(args.model),
            np.loadtxt(args.class_names, dtype=str, delimiter=","),
            cv2.imread(args.src),
        )
        exit(0)
