import argparse
import matplotlib
from plantcv import plantcv as pcv


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
    args = parser.parse_args()

    # Get options
    options = options(args.image)

    # Set debug to the global parameter
    pcv.params.debug = options.debug

    # Read image
    img, path, filename = pcv.readimage(filename=options.image)

    # Convert RGB to HSV and extract the saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")

    # Take a binary threshold to separate plant from background.
    s_thresh = pcv.threshold.binary(
        gray_img=s, threshold=85, max_value=255, object_type="light"
    )

    # Median Blur
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)

    # Gaussian blur
    s_gblur = pcv.gaussian_blur(img=s_mblur, ksize=(5, 5), sigma_x=0, sigma_y=None)

    # RGB -> LAB to extract the Blue channel
    b = pcv.rgb2gray_lab(rgb_img=img, channel="b")

    # Threshold the blue channel
    b_thresh = pcv.threshold.binary(
        gray_img=b, threshold=160, max_value=255, object_type="light"
    )

    # join the threshold saturation and blue-yellow images
    bs = pcv.logical_or(bin_img1=s_gblur, bin_img2=b_thresh)

    # Appy Mask
    masked = pcv.apply_mask(img=img, mask=bs, mask_color="white")

    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")

    # Threshold the green-magenta and blue images
    maskeda_thresh = pcv.threshold.binary(
        gray_img=masked_a, threshold=115, max_value=255, object_type="dark"
    )
    maskeda_thresh1 = pcv.threshold.binary(
        gray_img=masked_a, threshold=135, max_value=255, object_type="light"
    )
    maskedb_thresh = pcv.threshold.binary(
        gray_img=masked_b, threshold=128, max_value=255, object_type="light"
    )

    # Join the thresholded saturation and blue-yellow images (OR)
    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

    # Opening filters out bright noise from an image.
    opened_ab = pcv.opening(gray_img=ab)

    xor_img = pcv.logical_xor(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)

    # Fill small objects
    ab_fill = pcv.fill(bin_img=xor_img, size=200)

    # closing filters out dark noise from an image.

    closed_ab = pcv.closing(gray_img=ab_fill)

    # Apply mask (for vis images, mask_color=white)
    masked2 = pcv.apply_mask(img=masked, mask=closed_ab, mask_color="white")

    # Identify objects
    id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)

    # Define ROI
    roi1, roi_hierarchy = pcv.roi.rectangle(img=masked2, x=0, y=0, h=250, w=250)

    # Decide which objects to keep
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(
        img=img,
        roi_contour=roi1,
        roi_hierarchy=roi_hierarchy,
        object_contour=id_objects,
        obj_hierarchy=obj_hierarchy,
        roi_type="partial",
    )

    # Object combine kept objects
    obj, mask = pcv.object_composition(
        img=img, contours=roi_objects, hierarchy=hierarchy3
    )

    ### Analysis ###
    # Find shape properties, output shape image (optional)
    analysis_image = pcv.analyze_object(img=img, obj=obj, mask=mask, label="default")
