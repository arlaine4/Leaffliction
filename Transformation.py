import argparse
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from PIL import Image
import os
from tqdm import tqdm
import sys


def getlastname(path):
    if path.endswith("/"):
        return path.split("/")[-2].split(".")[0]
    return path.split("/")[-1].split(".")[0]


class Transformation:
    def __init__(self, options):
        self.options = options

        # apply write image
        pcv.params.debug_outdir = self.options.outdir
        if self.options.writeimg:
            self.name_save = self.options.outdir + "/" \
                + getlastname(self.options.image)

        # original
        self.img = None

        # gaussian_blur
        self.blur = None

        # masked
        self.masked2 = None
        self.ab = None

        # roi_objects
        self.roi_objects = None
        self.hierarchy = None
        self.kept_mask = None

        # analysis_objects
        self.mask = None
        self.obj = None

    def original(self):
        img, _, _ = pcv.readimage(filename=self.options.image)

        if self.options.debug == "print":
            pcv.print_image(
                img,
                filename=self.name_save + "_original.JPG",
            )

        self.img = img
        return img

    def gaussian_blur(self):
        if self.img is None:
            self.original()

        s = pcv.rgb2gray_hsv(rgb_img=self.img, channel="s")
        s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=60, max_value=255, object_type="light"
        )
        s_gblur = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5),
                                    sigma_x=0, sigma_y=None)

        if self.options.debug == "print":
            pcv.print_image(
                s_gblur,
                filename=self.name_save + "_gaussian_blur.JPG",
            )

        self.blur = s_gblur
        return s_gblur

    def masked(self):
        if self.blur is None:
            self.gaussian_blur()

        b = pcv.rgb2gray_lab(rgb_img=self.img, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b, threshold=200, max_value=255, object_type="light"
        )
        bs = pcv.logical_or(bin_img1=self.blur, bin_img2=b_thresh)

        masked = pcv.apply_mask(img=self.img, mask=bs, mask_color="white")

        masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")

        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, max_value=255,
            object_type="dark"
        )
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a, threshold=135, max_value=255,
            object_type="light"
        )
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b, threshold=128, max_value=255,
            object_type="light"
        )

        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

        xor_img = pcv.logical_xor(bin_img1=maskeda_thresh,
                                  bin_img2=maskedb_thresh)
        xor_img_color = pcv.apply_mask(img=self.img, mask=xor_img,
                                       mask_color="white")

        ab_fill = pcv.fill(bin_img=ab, size=200)

        masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color="white")

        if self.options.debug == "print":
            pcv.print_image(
                masked2,
                filename=self.name_save + "_masked.JPG",
            )
            pcv.print_image(
                xor_img_color,
                filename=self.name_save + "_xor.JPG",
            )

        self.masked2 = masked2
        self.ab = ab_fill
        return masked2

    def roi(self):
        if self.masked2 is None:
            self.masked()

        id_objects, obj_hierarchy = pcv.find_objects(img=self.img,
                                                     mask=self.ab)

        roi1, roi_hierarchy = pcv.roi.rectangle(img=self.img,
                                                x=0, y=0, h=250,
                                                w=250)

        pcv.params.debug = self.options.debug

        roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(
            img=self.img,
            roi_contour=roi1,
            roi_hierarchy=roi_hierarchy,
            object_contour=id_objects,
            obj_hierarchy=obj_hierarchy,
            roi_type="partial",
        )

        if self.options.debug == "print":
            file_rename = (
                self.options.outdir
                + "/"
                + str(pcv.params.device - 2)
                + "_obj_on_img.png"
            )
            file_delete = (
                self.options.outdir + "/" + str(pcv.params.device - 1)
                + "_roi_mask.png"
            )

            os.remove(file_delete)
            os.rename(file_rename, self.name_save + "_roi_mask.JPG")

        pcv.params.debug = None

        self.roi_objects = roi_objects
        self.hierarchy = hierarchy3
        self.kept_mask = kept_mask

        return roi_objects, hierarchy3, kept_mask, obj_area

    def analysis_objects(self):
        if self.roi_objects is None:
            self.roi()

        obj, mask = pcv.object_composition(
            img=self.img, contours=self.roi_objects, hierarchy=self.hierarchy
        )

        analysis_image = pcv.analyze_object(
            img=self.img, obj=obj, mask=mask, label="default"
        )

        if self.options.debug == "print":
            pcv.print_image(
                analysis_image,
                filename=self.name_save + "_analysis_objects.JPG",
            )

        self.mask = mask
        self.obj = obj
        return analysis_image

    def pseudolandmarks(self):
        if self.mask is None:
            self.analysis_objects()

        pcv.params.debug = self.options.debug

        top_x, bottom_x, center_v_x = pcv.x_axis_pseudolandmarks(
            img=self.img, obj=self.obj, mask=self.mask, label="default"
        )

        pcv.params.debug = None
        if self.options.debug == "print":
            file_rename = (
                self.options.outdir
                + "/"
                + str(pcv.params.device - 1)
                + "_x_axis_pseudolandmarks.png"
            )

            os.rename(file_rename, self.name_save + "_pseudolandmarks.JPG")
        return top_x, bottom_x, center_v_x

    def color_histogram(self):
        if self.mask is None:
            self.analysis_objects()

        color_histogram = pcv.analyze_color(
            rgb_img=self.img,
            mask=self.kept_mask,
            colorspaces="all",
            label="default",
        )

        if self.options.debug == "print":
            pcv.print_image(
                color_histogram,
                filename=self.name_save + "_color_histogram.JPG",
            )

        return color_histogram


class options:
    def __init__(
        self, path, debug="print", writeimg=True,
        result="results.json", outdir="."
    ):
        self.image = path
        self.debug = debug
        self.writeimg = writeimg
        self.result = result
        self.outdir = outdir
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)


def transform_image(options, training=False):
    """
    training mode will remove some transformations so
    we are not feeding the model useless pictures
    """
    transformation = Transformation(options)
    transformation.original()
    transformation.gaussian_blur()
    transformation.masked()
    transformation.roi()
    transformation.analysis_objects()
    transformation.pseudolandmarks()
    if not training:
        transformation.color_histogram()


def recalculate(src, path):
    if not src.endswith("/"):
        src += "/"

    last = getlastname(src)
    relative_path = path[len(src):]

    if relative_path == "":
        return last
    return last + "/" + relative_path


def already_done(path):
    if not os.path.isfile(path):
        return False
    return True


def batch_transform(src, dst, training=False):
    """
    Generate image transformation for each image found from the source
    directory.
    Training argument used when calling batch_transform before training
    a model.
    """
    if src is None or dst is None:
        raise Exception("Need to specify src and dst")
    if not os.path.isdir(src):
        raise Exception("src is not a dir")
    if not os.path.isdir(dst):
        os.makedirs(dst)

    for root, _, files in os.walk(src):
        name = recalculate(src, root)

        try:
            os.makedirs(os.path.join(dst, name))
        except FileExistsError:
            pass

        print("Doing batch for directory", name, "found",
              len(files), "pictures")
        for file in tqdm(files):
            if file.endswith(".JPG"):
                # if already_done(os.path.join(dst, name, file)):
                # continue
                opt = options(
                    os.path.join(root, file),
                    debug="print",
                    writeimg=True,
                    outdir=dst + "/" + name,
                )
                transform_image(opt, training)
        print()


def get_image(name, training=False):
    images = {}

    images["original"] = Image.open(name + "_original.JPG")
    images["gaussian_blur"] = Image.open(name +
                                         "_gaussian_blur.JPG") \
        .convert("RGB")
    images["masked"] = Image.open(name + "_masked.JPG")
    images["xor"] = Image.open(name + "_xor.JPG")
    images["analysis_objects"] = Image.open(name + "_analysis_objects.JPG")
    images["pseudolandmarks"] = Image.open(name + "_pseudolandmarks.JPG")
    if not training:
        images["color_histogram"] = Image.open(name + "_color_histogram.JPG")

    return images


def plot_images(options):
    name = options.outdir + "/" + getlastname(options.image)
    images = get_image(name)

    # plot all images
    fig, axs = plt.subplots(1, len(images), figsize=(12, 3))
    fig.suptitle(f"Image: {options.image}")
    plt.axis("off")
    axs = axs.flatten()
    for i, img, ax in zip(range(len(images)), images.values(), axs):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(list(images.keys())[i])
    plt.tight_layout()
    plt.show(block=True)


def transform_tmp(path):
    opt = options(path, outdir="./tmp")
    transform_image(opt, training=True)
    return get_image(opt.outdir + "/" + getlastname(opt.image), training=True)


def delete_tmp():
    os.system("rm -rf ./tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # check number of arguments
    if len(sys.argv) <= 2:
        parser.add_argument("img", type=str, help="Path to the image.")
    parser.add_argument("-src", type=str,
                        help="Path to the source dir or image.")
    parser.add_argument(
        "-dst", type=str,
        help="Path to the destination dir. (needed if src is a dir)"
    )
    parser.add_argument(
        "-t", "--training", action="store_true",
        help="Use this flag when calling batch_transform before training (default: False)"
    )
    args = parser.parse_args()

    if len(sys.argv) == 2 and os.path.isfile(args.img):
        if not args.img.endswith(".JPG"):
            exit("Not a JPG file")
        options = options(args.img, outdir="./tmp")
        transform_image(options)
        plot_images(options)
        delete_tmp
    else:
        batch_transform(args.src, args.dst, args.training)
