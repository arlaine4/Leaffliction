import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Transformation import getMasked, options

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction of images")
    parser.add_argument("image", type=str, help="Path to the image")
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

    # Load the model, image and class names
    model = load_model(args.model)
    image = cv2.imread(args.image)
    # Resizing image to match the input shape the model is expecting
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    class_names = np.loadtxt(args.class_names, dtype=str, delimiter=",")

    # Predict the class of the image
    # prediction = model.predict(np.expand_dims(image, axis=0))
    prediction = model.predict(image)
    class_name_prediction = class_names[np.argmax(prediction)]

    options = options(args.image, debug=None)
    masked = getMasked(options)

    plt.suptitle(f"Prediction: {class_name_prediction}")

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(masked)
    plt.axis("off")

    plt.show()
