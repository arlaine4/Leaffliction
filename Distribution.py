import os
import sys
import matplotlib.pyplot as plt
import random
import argparse


def load_images_from_directory(directory_path):
    images = os.listdir(directory_path)
    images = [image for image in images if image.endswith(".JPG")]
    return images


def main_images_distribution(directory_path):
    sub_directories = os.listdir(directory_path)
    valid_sub_directories = 0
    directory_images = {}
    total_images = 0
    for sub_directory in sub_directories:
        images = load_images_from_directory(os.path.join(directory_path, sub_directory))
        if len(images) != 0:
            total_images += len(images)
            directory_images[sub_directory] = images
            valid_sub_directories += 1
    print(
        f"Found {valid_sub_directories} sub directories in "
        f"{directory_path} with a total of {total_images} images."
    )
    return directory_images


def generate_random_hexa_color_codes(number_of_colors_to_generate):
    color_codes = []
    for i in range(number_of_colors_to_generate):
        color_codes.append("#" + ("%06x" % random.randint(0, 0xFFFFFF)))
    return color_codes


def plot_image_distribution(base_directory_name, images_with_directory_names):
    # Getting directories name
    labels = list(images_with_directory_names.keys())
    # Getting the number of images per directory
    images_list = list(images_with_directory_names.values())
    # Converting each list of images to the sum of itself
    total_images = [len(images) for images in images_list]
    colors = generate_random_hexa_color_codes(len(labels))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Images distribution in {base_directory_name}")

    # Pie chart
    plt.subplot(1, 2, 1)
    plt.pie(total_images, labels=labels, autopct="%1.1f%%", colors=colors)
    plt.title("Pie chart")

    # Bar chart
    plt.subplot(1, 2, 2)
    plt.bar(labels, total_images, color=colors)
    plt.grid(True)
    plt.title("Bar chart")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="Path to the source dir.")
    args = parser.parse_args()

    directory = args.src
    if not os.path.isdir(directory):
        sys.exit("Invalid directory path")
    directory_names_with_images = main_images_distribution(directory)
    plot_image_distribution(directory, directory_names_with_images)
