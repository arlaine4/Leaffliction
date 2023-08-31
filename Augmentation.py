import os
import sys
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from copy import deepcopy
from Distribution import load_images_from_directory

SAVING_PATH = ""


class ImageAugmentation:
    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        return image

    @staticmethod
    def save_image(image, method_name, custom_path=None, image_name=None):
        """
        After applying any augmentation method, save_image is called
        to save the augmented image as a copy of the original one
        """
        # This block is used if a folder is provided as a path argument
        if custom_path:
            custom_path += f"/{image_name.split('.')[0]}_{method_name}.JPG"
            if not os.path.exists(custom_path):
                cv2.imwrite(custom_path, image)
        # This block is used if the path points to an image
        else:
            save_path = SAVING_PATH.split("/")[:-1]
            image_name = SAVING_PATH.split("/")[-1].split(".")[0]
            destination_folder = "augmented_directory/" + "/".join(save_path[1:])
            if not os.path.isdir(destination_folder):
                os.makedirs(destination_folder)
            final_path = f"{destination_folder}/{image_name}_{method_name}.JPG"
            if not os.path.exists(final_path):
                cv2.imwrite(final_path, image)

    @staticmethod
    def rotate(image, angle=45, save_image=True):
        """
        Rotate image counter clock-wise to 'angle' value
        """
        image = imutils.rotate(image, angle)
        if save_image:
            ImageAugmentation.save_image(image, "rotate")
        return image

    @staticmethod
    def gaussian_blur(image, blur_value=(15, 15), save_image=True):
        """
        Simple image blur that aims to reduce noise in the picture
        """
        image = cv2.GaussianBlur(image, blur_value, 0)
        if save_image:
            ImageAugmentation.save_image(image, "gaussian_blur")
        return image

    @staticmethod
    def contrast(image, alpha=1.5, beta=2, save_image=True):
        """
        Changing brightness and contrast values to help the
        model in dealing with luminosity variations.
        """
        image = cv2.convertScaleAbs(image, alpha, beta)
        if save_image:
            ImageAugmentation.save_image(image, "contrast")
        return image

    @staticmethod
    def reflection(image, save_image=True):
        """
        Flipping image upside down, as if the leaf was looking
        at it's reflection from the water of a lake for example
        """
        rows, cols, dim = image.shape
        matrix = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
        image = cv2.warpPerspective(image, matrix, (cols, rows))
        if save_image:
            ImageAugmentation.save_image(image, "reflection")
        return image

    @staticmethod
    def scaling(image, scale_factor=0.75, save_image=True):
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
        width_points = [
            center_x - int(center_x * scale_factor),
            center_x + int(center_x * scale_factor),
        ]
        height_points = [
            center_y - int(center_y * scale_factor),
            center_y + int(center_y * scale_factor),
        ]

        # Applying the crop
        image = image[
            width_points[0] : width_points[1], height_points[0] : height_points[1]
        ]
        # Resizing the image back to its original
        # dimensions with cropping applied
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if save_image:
            ImageAugmentation.save_image(image, "scaling")
        return image

    @staticmethod
    def shear(image, save_image=True):
        """
        Simulating a different angle of view, POV change so to say
        """
        rows, cols, dims = image.shape
        matrix = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
        image = cv2.warpPerspective(image, matrix, (int(cols * 1.5), int(rows * 1.5)))
        if save_image:
            ImageAugmentation.save_image(image, "shear")
        return image


def apply_augmentation_techniques(image, image_augmentation, save_image=True, training=False):
    if training:
        methods = ["reflection", "scaling"]
    else:
        methods = ["reflection", "scaling", "rotate", "gaussian_blur", "contrast", "shear"]
    images = [image]
    for method in methods:
        function_call = getattr(image_augmentation, method)
        images.append(function_call(image, save_image=save_image))
    methods.insert(0, "original")
    return methods, images


def plot_all_pictures(image, image_path, image_augmentation):
    methods, images = apply_augmentation_techniques(image, image_augmentation)

    fig, axs = plt.subplots(1, len(methods), figsize=(12, 3))
    fig.suptitle(f"Image: {image_path}")
    plt.axis("off")
    axs = axs.flatten()
    for i, img, ax in zip(range(len(methods)), images, axs):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(methods[i])
    plt.tight_layout()
    plt.show()


def main_augmentation(path, mode, training=False):
    # Removing useless \ inside path because of bad image name formatting
    path.replace("\\", "") if "\\" in path else path
    img_augmentation = ImageAugmentation()
    # Processing single image path
    if mode == "image":
        global SAVING_PATH
        SAVING_PATH = path
        image = img_augmentation.load_image(path)
        plot_all_pictures(image, path, img_augmentation)
    # Processing folder path
    else:
        # Dict of folder paths and their corresponding files
        final_dirs = {}
        # Extracing only 'final' dirs, ones that only have files inside and not
        # more sub folders
        for root, dirs, files in os.walk(path):
            if not dirs:
                final_dirs[root] = load_images_from_directory(root)
        # Running augmentation loop for each of the final folders found
        for directory, items in final_dirs.items():
            print(
                f"Doing batch for directory {directory}," f"found {len(items)} pictures"
            )
            # Generating final destination path in augmented_directory
            new_d_name_augmented = "/".join(directory.split("/")[1:])
            try:
                os.makedirs(os.path.join("augmented_directory", new_d_name_augmented))
            except FileExistsError:
                pass
            # Running image augmentation for each image found inside the folder
            for image in tqdm(items):
                image_name = deepcopy(image)
                image = img_augmentation.load_image(os.path.join(directory, image))
                methods, images = apply_augmentation_techniques(
                    image, img_augmentation, save_image=False, training=training
                )
                for i, img in enumerate(images):
                    img_augmentation.save_image(
                        img,
                        methods[i],
                        os.path.join("augmented_directory", new_d_name_augmented),
                        image_name,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help="Path to the image or folder containing images to augment"
    )
    args = parser.parse_args()

    default_path = args.path
    # Checking if provided path points to an image or a folder
    if not os.path.exists(default_path):
        raise FileNotFoundError("Path doesn't exist")
    if os.path.isfile(default_path):
        if not "".join(default_path.split("/")[-1]).endswith(".JPG"):
            raise FileNotFoundError("Invalid file extension")
        main_augmentation(default_path, "image")
    elif os.path.isdir(default_path):
        main_augmentation(default_path, "batch")
