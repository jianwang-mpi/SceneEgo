# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# References:
# - https://github.com/xiumingzhang/GenRe-ShapeHD

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch


class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.RandomBackground(),
    >>>     transforms.CenterCrop(127, 127, 3),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bounding_box=None):
        for t in self.transforms:
            if t.__class__.__name__ == 'RandomCrop' or t.__class__.__name__ == 'CenterCrop':
                image = t(image, bounding_box)
            else:
                image = t(image)

        return image


class ToTensor(object):
    """
    Convert a numpy.ndarray to tensor.
    """

    def __call__(self, image):
        assert (isinstance(image, np.ndarray))
        # HWC to CHW
        array = np.transpose(image, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        return tensor.float()


class BGR2RGB:
    """
    convert BGR image to RGB image
    """

    def __call__(self, image):
        # BGR to RGB
        return image[:, :, ::-1]


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        assert (isinstance(image, np.ndarray))
        image -= self.mean
        image /= self.std

        return image


class SimpleNormalize(object):
    def __init__(self):
        self.mean = 0.4

    def __call__(self, image):
        assert (isinstance(image, np.ndarray))
        image -= self.mean
        return image


class RandomPermuteRGB(object):
    """
    Random permute RGB channels???
    """

    def __call__(self, image):
        assert (isinstance(image, np.ndarray))

        random_permutation = np.random.permutation(3)
        image = image[..., random_permutation]

        return image


class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, image, bounding_box=None):

        img_height, img_width, _ = image.shape

        if bounding_box is not None:
            bounding_box = [
                bounding_box[0],
                bounding_box[1],
                bounding_box[2],
                bounding_box[3]
            ]  # yapf: disable

            # Calculate the size of bounding boxes
            bbox_width = bounding_box[2] - bounding_box[0]
            bbox_height = bounding_box[3] - bounding_box[1]
            bbox_x_mid = (bounding_box[2] + bounding_box[0]) * .5
            bbox_y_mid = (bounding_box[3] + bounding_box[1]) * .5

            # Make the crop area as a square
            square_object_size = max(bbox_width, bbox_height)

            x_left = int(bbox_x_mid - square_object_size * .5)
            x_right = int(bbox_x_mid + square_object_size * .5)
            y_top = int(bbox_y_mid - square_object_size * .5)
            y_bottom = int(bbox_y_mid + square_object_size * .5)

            # If the crop position is out of the image, fix it with padding
            pad_x_left = 0
            if x_left < 0:
                pad_x_left = -x_left
                x_left = 0
            pad_x_right = 0
            if x_right >= img_width:
                pad_x_right = x_right - img_width + 1
                x_right = img_width - 1
            pad_y_top = 0
            if y_top < 0:
                pad_y_top = -y_top
                y_top = 0
            pad_y_bottom = 0
            if y_bottom >= img_height:
                pad_y_bottom = y_bottom - img_height + 1
                y_bottom = img_height - 1

            # Padding the image and resize the image
            processed_image = np.pad(
                image[y_top:y_bottom + 1, x_left:x_right + 1], ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right),
                                                                (0, 0)),
                mode='edge')
            processed_image = cv2.resize(processed_image, (self.img_size_w, self.img_size_h))
        else:
            if img_height > self.crop_size_h and img_width > self.crop_size_w:
                x_left = int(img_width - self.crop_size_w) // 2
                x_right = int(x_left + self.crop_size_w)
                y_top = int(img_height - self.crop_size_h) // 2
                y_bottom = int(y_top + self.crop_size_h)
            else:
                x_left = 0
                x_right = img_width
                y_top = 0
                y_bottom = img_height

            processed_image = cv2.resize(image[y_top:y_bottom, x_left:x_right], (self.img_size_w, self.img_size_h))

        return processed_image


class RandomCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, image, silhouette=None, bounding_box=None):

        img_height, img_width, crop_size_c = image.shape

        if bounding_box is not None:
            bounding_box = [
                bounding_box[0],
                bounding_box[1],
                bounding_box[2],
                bounding_box[3]
            ]  # yapf: disable

            # Calculate the size of bounding boxes
            bbox_width = bounding_box[2] - bounding_box[0]
            bbox_height = bounding_box[3] - bounding_box[1]
            bbox_x_mid = (bounding_box[2] + bounding_box[0]) * .5
            bbox_y_mid = (bounding_box[3] + bounding_box[1]) * .5

            # Make the crop area as a square
            square_object_size = max(bbox_width, bbox_height)
            square_object_size = square_object_size * random.uniform(0.8, 1.2)

            x_left = int(bbox_x_mid - square_object_size * random.uniform(.4, .6))
            x_right = int(bbox_x_mid + square_object_size * random.uniform(.4, .6))
            y_top = int(bbox_y_mid - square_object_size * random.uniform(.4, .6))
            y_bottom = int(bbox_y_mid + square_object_size * random.uniform(.4, .6))

            # If the crop position is out of the image, fix it with padding
            pad_x_left = 0
            if x_left < 0:
                pad_x_left = -x_left
                x_left = 0
            pad_x_right = 0
            if x_right >= img_width:
                pad_x_right = x_right - img_width + 1
                x_right = img_width - 1
            pad_y_top = 0
            if y_top < 0:
                pad_y_top = -y_top
                y_top = 0
            pad_y_bottom = 0
            if y_bottom >= img_height:
                pad_y_bottom = y_bottom - img_height + 1
                y_bottom = img_height - 1

            # Padding the image and resize the image
            processed_image = np.pad(
                image[y_top:y_bottom + 1, x_left:x_right + 1], ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right),
                                                                (0, 0)), mode='edge')
            processed_image = cv2.resize(processed_image, (self.img_size_w, self.img_size_h))
            if silhouette is not None:
                processed_silhouette = np.pad(
                    silhouette[y_top:y_bottom + 1, x_left:x_right + 1],
                    ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right),
                     (0, 0)), mode='constant')
                processed_silhouette = cv2.resize(processed_silhouette, (self.img_size_w, self.img_size_h))
            else:
                processed_silhouette = None


        else:
            if img_height > self.crop_size_h and img_width > self.crop_size_w:
                x_left = int(img_width - self.crop_size_w) // 2
                x_right = int(x_left + self.crop_size_w)
                y_top = int(img_height - self.crop_size_h) // 2
                y_bottom = int(y_top + self.crop_size_h)
            else:
                x_left = 0
                x_right = img_width
                y_top = 0
                y_bottom = img_height

            processed_image = cv2.resize(image[y_top:y_bottom, x_left:x_right], (self.img_size_w, self.img_size_h))
            if silhouette is not None:
                processed_silhouette = cv2.resize(silhouette[y_top:y_bottom, x_left:x_right],
                                                  (self.img_size_w, self.img_size_h))
            else:
                processed_silhouette = None

        return processed_image, processed_silhouette


class RandomFlip(object):
    def __call__(self, image, silhouette=None):
        assert (isinstance(image, np.ndarray))

        if random.randint(0, 1):
            image = np.fliplr(image)
            if silhouette is not None:
                silhouette = np.fliplr(silhouette)

        return image, silhouette


class RandomNoise(object):
    def __init__(self,
                 noise_std,
                 eigvals=(0.2175, 0.0188, 0.0045),
                 eigvecs=((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203))):
        self.noise_std = noise_std
        self.eigvals = np.array(eigvals)
        self.eigvecs = np.array(eigvecs)

    def __call__(self, image):
        alpha = np.random.normal(loc=0, scale=self.noise_std, size=3)
        noise_rgb = \
            np.sum(
                np.multiply(
                    np.multiply(
                        self.eigvecs,
                        np.tile(alpha, (3, 1))
                    ),
                    np.tile(self.eigvals, (3, 1))
                ),
                axis=1
            )

        # Allocate new space for storing processed images
        img_height, img_width, img_channels = image.shape
        assert (img_channels == 3), "Please use RandomBackground to normalize image channels"
        for i in range(img_channels):
            image[:, :, i] += noise_rgb[i]

        return image


class ColorJitter(object):
    def __init__(self, color_add, color_mul):
        self.color_add_low = 0
        self.color_add_high = color_add
        self.color_mul_low = 1 - color_mul
        self.color_mul_high = 1 + color_mul

    def __call__(self, rendering_image):
        color_add = np.random.uniform(self.color_add_low, self.color_add_high, size=(1, 1, 3))
        color_mul = np.random.uniform(self.color_mul_low, self.color_mul_high, size=(1, 1, 3))
        rendering_image = rendering_image + color_add
        rendering_image = rendering_image * color_mul
        return rendering_image


if __name__ == '__main__':
    from config import consts
    import skimage.io as img_io

    # test random noise
    random_noise = RandomNoise(consts.img.noise_std)
    image = img_io.imread('/home/wangjian/Develop/3DReconstruction/tmp/bobo.jpg') / 255.
    # img_io.imshow(image)
    # img_io.show()
    image = random_noise(image)
    image = image / np.max(image, (0, 1))
    img_io.imshow(image)
    img_io.show()
