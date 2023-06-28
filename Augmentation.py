import os
import sys
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt


class ImageAugmentation:
    @staticmethod
    def load_image(image_path):
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        return image

    @staticmethod
    def rotate(image, angle=45):
        return imutils.rotate(image, angle)

    @staticmethod
    def gaussian_blur(image, blur_value=(15, 15)):
        return cv2.GaussianBlur(image, blur_value, 0)

    @staticmethod
    def contrast(image, alpha=1.5, beta=2):
        return cv2.convertScaleAbs(image, alpha, beta)

    @staticmethod
    def reflection(image):
        rows, cols, dim = image.shape
        matrix = np.float32([[1, 0, 0],
                             [0, -1, rows],
                             [0, 0, 1]])
        return cv2.warpPerspective(image, matrix, (cols, rows))

    @staticmethod
    def scaling(image, scale_factor=.3):
        # Pas sur de laisser celle la
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # return image.resize((int(image.shape[1] / scale_factor),
        #                     int(image.shape[0] / scale_factor)),
        #                     refcheck=False)

    @staticmethod
    def shear(image):
        rows, cols, dims = image.shape
        matrix = np.float32([[1, 0.5, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        return cv2.warpPerspective(image, matrix, (int(cols * 1.5),
                                                   int(rows * 1.5)))


def main_augmentation(image_path):
    img_augmentation = ImageAugmentation()
    image_path.replace('\\', '')
    image = img_augmentation.load_image(image_path)

    reflection_image = img_augmentation.reflection(image)
    scaled_image = img_augmentation.scaling(image)
    rotated_image = img_augmentation.rotate(image)
    blurred_image = img_augmentation.gaussian_blur(image)
    contrast_image = img_augmentation.contrast(image)
    sheared_image = img_augmentation.shear(image)
    # temporary stuff under there,
    # will probably change the scaled augmentation technique by another one,
    # I'm not convinced by this one
    plt.imshow(sheared_image)
    plt.show()
    plt.imshow(reflection_image)
    plt.show()
    plt.imshow(scaled_image)
    plt.show()
    plt.imshow(blurred_image)
    plt.show()
    plt.imshow(rotated_image)
    plt.show()
    plt.imshow(contrast_image)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Missing image path')
    path = sys.argv[1]
    if not os.path.isfile(path) or \
            not ''.join(path.split('/')[-1]).endswith('.JPG'):
        sys.exit('File does not exists or has invalid extension')
    main_augmentation(path)
