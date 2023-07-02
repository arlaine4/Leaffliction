import argparse
import matplotlib as plt
from plantcv import plantcv as pcv
import os
import sys


class Transformation:
    def __init__(self, options):
        self.options = options

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

        if self.options.debug == "plot":
            pcv.plot_image(img)

        self.img = img
        return img

    def gaussian_blur(self):
        if self.img is None:
            raise Exception("Need to call original() first")

        s = pcv.rgb2gray_hsv(rgb_img=self.img, channel="s")
        s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=60, max_value=255, object_type="light"
        )
        s_gblur = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)

        if self.options.debug == "plot":
            pcv.plot_image(s_gblur)

        self.blur = s_gblur
        return s_gblur

    def masked(self):
        if self.blur is None:
            raise Exception("Need to call gaussian_blur() first")

        b = pcv.rgb2gray_lab(rgb_img=self.img, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b, threshold=200, max_value=255, object_type="light"
        )
        bs = pcv.logical_or(bin_img1=self.blur, bin_img2=b_thresh)

        masked = pcv.apply_mask(img=self.img, mask=bs, mask_color="white")

        masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")

        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, max_value=255, object_type="dark"
        )
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a, threshold=135, max_value=255, object_type="light"
        )
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b, threshold=128, max_value=255, object_type="light"
        )

        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

        opened_ab = pcv.opening(gray_img=ab)

        xor_img = pcv.logical_xor(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)

        ab_fill = pcv.fill(bin_img=ab, size=200)

        closed_ab = pcv.closing(gray_img=ab_fill)

        masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color="white")

        if self.options.debug == "plot":
            pcv.plot_image(masked2)

        self.masked2 = masked2
        self.ab = ab_fill
        return masked2

    def roi(self):
        if self.masked2 is None:
            raise Exception("Need to call mask() first")

        id_objects, obj_hierarchy = pcv.find_objects(img=self.img, mask=self.ab)

        roi1, roi_hierarchy = pcv.roi.rectangle(img=self.img, x=0, y=0, h=250, w=250)

        if self.options.debug == "plot":
            pcv.params.debug = "plot"

        roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(
            img=self.img,
            roi_contour=roi1,
            roi_hierarchy=roi_hierarchy,
            object_contour=id_objects,
            obj_hierarchy=obj_hierarchy,
            roi_type="partial",
        )

        if self.options.debug == "plot":
            pcv.params.debug = None

        self.roi_objects = roi_objects
        self.hierarchy = hierarchy3
        self.kept_mask = kept_mask

        return roi_objects, hierarchy3, kept_mask, obj_area

    def analysis_objects(self):
        if self.roi_objects is None:
            raise Exception("Need to call mask() first")

        obj, mask = pcv.object_composition(
            img=self.img, contours=self.roi_objects, hierarchy=self.hierarchy
        )

        analysis_image = pcv.analyze_object(
            img=self.img, obj=obj, mask=mask, label="default"
        )

        if self.options.debug == "plot":
            pcv.plot_image(analysis_image)

        self.mask = mask
        self.obj = obj
        return analysis_image

    def peudolandmarks(self):
        if self.mask is None:
            raise Exception("Need to call analysis_objects() first")

        if self.options.debug == "plot":
            pcv.params.debug = "plot"

        top_x, bottom_x, center_v_x = pcv.x_axis_pseudolandmarks(
            img=self.img, obj=self.obj, mask=self.mask, label="default"
        )

        if self.options.debug == "plot":
            pcv.params.debug = None
        return top_x, bottom_x, center_v_x

    def color_histogram(self):
        if self.mask is None:
            raise Exception("Need to call analysis_objects() first")

        color_histogram = pcv.analyze_color(
            rgb_img=self.img,
            mask=self.kept_mask,
            colorspaces="all",
            label="default",
        )

        if self.options.debug == "plot":
            pcv.plot_image(color_histogram)

        return color_histogram


class options:
    def __init__(
        self, path, debug="plot", writeimg=False, result="results.json", outdir="."
    ):
        self.image = path
        self.debug = debug
        self.writeimg = writeimg
        self.result = result
        self.outdir = outdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to input image file.")
    parser.add_argument("-src", type=str, help="Path to the source dir.")
    parser.add_argument("-dst", type=str, help="Path to the destination dir.")
    args = parser.parse_args()

    if args.image is not None:
        options = options(args.image)
        transformation = Transformation(options)
        transformation.original()
        transformation.gaussian_blur()
        transformation.masked()
        transformation.roi()
        transformation.analysis_objects()
        transformation.peudolandmarks()
        transformation.color_histogram()
    else:
        if args.src is None or args.dst is None:
            raise Exception("Need to specify src and dst")

        src = args.src
        dst = args.dst

        # TODO SAVE RESULTS

        for root, dirs, files in os.walk(src):
            for file in files:
                if file.endswith(".JPG") or file.endswith(".jpg"):
                    options = options(os.path.join(root, file), debug=None)
                    transformation = Transformation(options)
                    transformation.original()
                    transformation.gaussian_blur()
                    transformation.masked()
                    transformation.roi()
                    transformation.analysis_objects()
                    transformation.peudolandmarks()
                    transformation.color_histogram()
