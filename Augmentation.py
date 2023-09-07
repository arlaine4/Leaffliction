import os
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import argparse
import shutil
from tqdm import tqdm
from copy import deepcopy

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
            destination_folder = "augmented_directory/" \
                + "/".join(save_path[1:])
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
            width_points[0]: width_points[1],
            height_points[0]: height_points[1]
        ]
        # Resizing the image back to its original
        # dimensions with cropping applied
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA)
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
        image = cv2.warpPerspective(image, matrix,
                                    (int(cols * 1.5), int(rows * 1.5)))
        if save_image:
            ImageAugmentation.save_image(image, "shear")
        return image


def apply_augmentation_techniques(image, image_augmentation, save_image=True,
                                  training=False):
    if training:
        methods = ["reflection", "scaling", "contrast"]
    else:
        methods = ["reflection", "scaling", "rotate",
                   "gaussian_blur", "contrast", "shear"]
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


def main_augmentation(path, mode, training=False,
                      augmentation_folder_path='augmented_directory'):
    # Removing useless \ inside path because of bad image name formatting
    path.replace('\\', '') if '\\' in path else path
    img_augmentation = ImageAugmentation()
    if mode == "image":
        global SAVING_PATH
        SAVING_PATH = path
        img = img_augmentation.load_image(path)
        plot_all_pictures(img, path, img_augmentation)
    else:
        # Iterating from base directory,
        # getting path for every 'final sub directory'
        final_dirs = {}
        for root, dirs, files in os.walk(path):
            if not dirs:
                final_dirs[root] = len(files)
        # List of folder keys to be augmented because the image
        # count is below the mean
        to_augment = []
        # The folder with the lowest number of pics will be the goal.
        # e.g : 275 images, * 6 because we have 6 image augmentation
        # techniques available, so the augmentation goal towards
        # every folder will be 1650 images.
        to_augment_goal = min(final_dirs.values())
        to_augment_goal *= 6
        for key, nb_img in final_dirs.items():
            if nb_img < to_augment_goal:
                to_augment.append(key)

        if not os.path.isdir(augmentation_folder_path):
            os.makedirs(augmentation_folder_path)
        for folder in to_augment:
            generation_goal = to_augment_goal - final_dirs[folder]
            print(f'Starting augmentation for folder {folder}')
            print(f'Starting from {final_dirs[folder]}'
                  f' images towards {to_augment_goal}')
            print(f'Will generate {generation_goal} images')
            augmented_dir_name = folder.split('/')[-1]
            try:
                os.makedirs(os.path.join(augmentation_folder_path,
                                         augmented_dir_name))
            except FileExistsError:
                pass

            # Listing images in the directory that needs augmentation
            images = os.listdir(folder)
            for img in images:
                cv2.imwrite(os.path.join(augmentation_folder_path,
                                         augmented_dir_name, img),
                            img_augmentation.load_image(os.path.join(folder,
                                                                     img)))
            # Counting after each image augmentation until the count
            # reaches the to_augment_goal
            count = 0

            # compute number of aumentations needed by images
            needed_by_images = int(generation_goal / len(images)) + 1

            for image in tqdm(images):
                if count == generation_goal:
                    break
                image_name = deepcopy(image)
                image = img_augmentation.load_image(os.path.join(folder,
                                                                 image_name))
                # Generating the augmented images matrix
                methods, imgs = apply_augmentation_techniques(
                    image, img_augmentation, save_image=False,
                    training=training
                )

                # remove original image from methods and imgs
                methods = methods[1:]
                imgs = imgs[1:]

                methods_random = np.random.choice(methods, needed_by_images, replace=False)

                # Iterating over all the augmented images generated.
                # Saving the augmented images until we reach the
                # to_augment_goal
                for i, img in enumerate(imgs):
                    if methods[i] not in methods_random:
                        continue
                    count += 1
                    if count == generation_goal:
                        break
                    img_augmentation.save_image(
                        img,
                        methods[i],
                        os.path.join(augmentation_folder_path,
                                     augmented_dir_name),
                        image_name
                    )

            if count != generation_goal:
                raise ValueError("Impossible to generate enough images")



def split_for_test_set(split):
    """
    Splitting the augmented images into a test set.
    in the dir augmented_directory_test
    """
    print("Splitting {} of the augmented images for test set".format(split))
    augmented_dir = "augmented_directory"
    test_dir = "augmented_directory_test"
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    for root, dirs, files in os.walk(augmented_dir):
        if not dirs:
            test_dir_name = root.split('/')[-1]
            try:
                os.makedirs(os.path.join(test_dir, test_dir_name))
            except FileExistsError:
                pass
            images = os.listdir(root)
            np.random.shuffle(images)
            split_index = int(len(images) * split)
            for image in images[:split_index]:
                # copy in augmented_directory_test and remove from augmented_directory
                shutil.copy(os.path.join(root, image),
                            os.path.join(test_dir, test_dir_name, image))
                os.remove(os.path.join(root, image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help="Path to the image or folder containing images to augment"
    )
    parser.add_argument("-s", "--split",
                        help="Create a test split (default 0.1)",
                        type=float, default=0.1)
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
        # if augmented_diorectory alredy exit rm
        if os.path.isdir("augmented_directory"):
            shutil.rmtree("augmented_directory")
        main_augmentation(default_path, "batch")
        if args.split > 0:
            if os.path.isdir("augmented_directory_val"):
                shutil.rmtree("augmented_directory_val")
            split_for_test_set(args.split)
