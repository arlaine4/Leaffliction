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
        """
        Rotate image counter clock-wise to 'angle' value
        """
        image = imutils.rotate(image, angle)
        ImageAugmentation.save_image(image, 'rotate')
        return image

    @staticmethod
    def gaussian_blur(image, blur_value=(15, 15)):
        """
        Simple image blur that aims to reduce noise in the picture
        """
        image = cv2.GaussianBlur(image, blur_value, 0)
        ImageAugmentation.save_image(image, 'gaussian_blur')
        return image

    @staticmethod
    def contrast(image, alpha=1.5, beta=2):
        """
        Changing brightness and contrast values to help the
        model in dealing with luminosity variations.
        """
        image = cv2.convertScaleAbs(image, alpha, beta)
        ImageAugmentation.save_image(image, 'contrast')
        return image

    @staticmethod
    def reflection(image):
        """
        Flipping image upside down, as if the leaf was looking
        at it's reflection from the water of a lake for example
        """
        rows, cols, dim = image.shape
        matrix = np.float32([[1, 0, 0],
                             [0, -1, rows],
                             [0, 0, 1]])
        image = cv2.warpPerspective(image, matrix, (cols, rows))
        ImageAugmentation.save_image(image, 'reflection')
        return image

    @staticmethod
    def scaling(image, scale_factor=.75):
        """
        Cropping outside pixels, very useful in our case
        because most of the images will have the leaf in the
        middle or close to the middle of the image.
        With this outside pixel removal we provide only relevant pixel
        values to the classification model
        """
        # Extracting base image shape information
        width, height = image.shape[:2]
        # Determining the center for x and y coordinates
        center_x, center_y = width // 2, height // 2
        # Defining starting and ending points from before and after the center
        # for each axis, start_point -> center_axis <- end_point
        # the goal is to only remove the outside pixels
        width_points = [center_x - int(center_x * scale_factor), center_x + int(center_x * scale_factor)]
        height_points = [center_y - int(center_y * scale_factor), center_y + int(center_y * scale_factor)]

        # Applying the crop
        image = image[width_points[0]:width_points[1], height_points[0]: height_points[1]]
        # Resizing the image back to it's original dimensions with cropping applied
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        ImageAugmentation.save_image(image, "scaling")
        return image

    @staticmethod
    def save_image(image, method_name):
        """
        After applying any augmentation method, save_image is called
        to save the augmented image as a copy of the original one
        """
        save_path = SAVING_PATH.split('/')[:-1]
        image_name = SAVING_PATH.split('/')[-1].split('.')[0]
        final_path = '/'.join(save_path) + '/' + image_name + f'_{method_name}.JPG'
        cv2.imwrite(final_path, image)

    @staticmethod
    def shear(image):
        """
        Simulating a different angle of view, POV change so to say
        """
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
    # MAKE AUGMENTED_DIRECTORY and save the pictures there
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
