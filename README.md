# Leaffliction 🌿

## 📝 Description

This is a computer vision project for plant leaf diseases.
In this project we'll be doing image dataset analysis, data augmentation, image transformations and image classification.

## 📦 Installation

To setup the project, you need to launch the following command:

```bash
git clone https://github.com/arlaine4/Leaffliction && cd Leaffliction
bash setup.sh
source venv/bin/activate
```

## 📑 Summary

- [Data Analysis](#-data-analysis)
- [Data Augmentation](#-data-augmentation)
- [Image Transformations](#-image-transformations)
- [Classification](#-classification)

## 🧐 Data analysis

A program named **Distribution.py** is created to extract and analyze the image dataset of plant leaves. Pie charts and Bar charts are generated for each plant type, using images available in the subdirectories of the given input directory.

![image](https://i.imgur.com/KafKy94.png)

## ➕ Data Augmentation

To balance the data set, we have a second program called **Augmentation.py**. It uses data augmentation techniques such as rotating, flipping, cropping, etc. To create 6 types of augmented images for each original image.

![image](https://i.imgur.com/pjyLTn8.png)

## 🖼️ Image Transformation

In this part, the **Transformation.py** program is created to directly extract features from plant leaf images. Transformations like Gaussian blur, ROI (Region of Interest) objects, object analysis, etc., are applied to images to facilitate key information extraction. For this part we use the [PlantCV](https://plantcv.readthedocs.io/en/stable/) library.

![image](https://i.imgur.com/YsMiy6B.gif)

## 🤖 Classification

The final step involves developing two programs: **train.py** and **predict.py**.

The **train** program uses augmented images to learn the characteristics of specified leaf diseases, utilizing a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) through **Keras**. The learning results are saved and returned in a .zip archive.

The **predict** program takes a leaf image as input, displays it along with its transformations, and predicts the type of disease specified in the leaf.
