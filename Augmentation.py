import os
import sys
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt


SAVING_PATH = ''


class ImageAugmentation:
    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        return image

    @staticmethod
    def rotate(image, angle=45):
        image = imutils.rotate(image, angle)
        ImageAugmentation.save_image(image, 'rotate')
        return image

    @staticmethod
    def gaussian_blur(image, blur_value=(15, 15)):
        image = cv2.GaussianBlur(image, blur_value, 0)
        ImageAugmentation.save_image(image, 'gaussian_blur')
        return image

    @staticmethod
    def contrast(image, alpha=1.5, beta=2):
        image = cv2.convertScaleAbs(image, alpha, beta)
        ImageAugmentation.save_image(image, 'contrast')
        return image

    @staticmethod
    def reflection(image):
        rows, cols, dim = image.shape
        matrix = np.float32([[1, 0, 0],
                             [0, -1, rows],
                             [0, 0, 1]])
        image = cv2.warpPerspective(image, matrix, (cols, rows))
        ImageAugmentation.save_image(image, 'reflection')
        return image

    @staticmethod
    def scaling(image, scale_factor=.3):
        # A CHANGER
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # ImageAugmentation.save_image(image, 'scaling')
        return image

    @staticmethod
    def save_image(image, method_name):
        save_path = SAVING_PATH.split('/')[:-1]
        image_name = SAVING_PATH.split('/')[-1].split('.')[0]
        final_path = '/'.join(save_path) + '/' + image_name + f'_{method_name}.JPG'
        cv2.imwrite(final_path, image)

    @staticmethod
    def shear(image):
        rows, cols, dims = image.shape
        matrix = np.float32([[1, 0.5, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        image = cv2.warpPerspective(image, matrix, (int(cols * 1.5),
                                                    int(rows * 1.5)))
        ImageAugmentation.save_image(image, "shear")
        return image


def plot_all_pictures(image, image_path, image_augmentation):
    methods = ['reflection', 'scaling', 'rotate', 'gaussian_blur', 'contrast', 'shear']
    images = [image]
    for method in methods:
        function_call = getattr(image_augmentation, method)
        images.append(function_call(image))

    methods.insert(0, 'original')
    fig, axs = plt.subplots(1, len(methods), figsize=(12, 3))
    plt.axis('off')
    axs = axs.flatten()
    for i, img, ax in zip(range(len(methods)), images, axs):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(methods[i])
    plt.show()


def main_augmentation(image_path):
    global SAVING_PATH
    img_augmentation = ImageAugmentation()
    image_path.replace('\\', '')
    SAVING_PATH = image_path
    image = img_augmentation.load_image(image_path)

    plot_all_pictures(image, image_path, img_augmentation)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Missing image path')
    path = sys.argv[1]
    if not os.path.isfile(path) or \
            not ''.join(path.split('/')[-1]).endswith('.JPG'):
        sys.exit('File does not exists or has invalid extension')
    main_augmentation(path)
