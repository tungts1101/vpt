import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from PIL import ImageOps, Image, ImageFilter
import numpy as np


class ResizedRandomCrop(object):
    def __init__(
        self,
        size=(224, 224),
        scale=(0.08, 1.0),
        image_interpolation=Image.BICUBIC,
        mask_interpolation=Image.NEAREST,
    ):
        self.size = size
        self.scale = scale
        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation

    def __call__(self, sample):
        image, mask = sample
        scale = random.uniform(*self.scale)
        original_width, original_height = image.size
        crop_width = int(scale * original_width)
        crop_height = int(scale * original_height)

        # Calculate crop area and perform crop
        i, j, h, w = transforms.RandomCrop.get_params(
            img=image, output_size=(crop_height, crop_width)
        )
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        # Resize cropped to desired size
        image = F.resize(image, self.size, self.image_interpolation)
        mask = F.resize(mask, self.size, self.mask_interpolation)

        return image, mask


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class ColorTransform(object):
    def __init__(self):
        # code adopted from https://github.com/facebookresearch/vicreg/blob/main/augmentations.py
        self.transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
            ]
        )

    def __call__(self, sample):
        image, mask = sample
        return self.transform(image), mask


class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample
        return F.to_tensor(image), F.to_tensor(mask).long()


class Normalization(object):
    def __call__(self, sample):
        image, mask = sample
        return (
            F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            mask,
        )


class FixedResize(object):
    def __init__(
        self,
        size=224,
        image_interpolation=Image.BICUBIC,
        mask_interpolation=Image.NEAREST,
    ):
        self.size = size
        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation

    def __call__(self, sample):
        image, mask = sample
        image = F.resize(image, self.size, self.image_interpolation)
        mask = F.resize(mask, self.size, self.mask_interpolation)
        return image, mask

class BarlowTwinTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample):
        y1 = self.transform(sample[0])
        y2 = self.transform_prime(sample[0])
        return y1, y2


def get_train_transformation():
    return transforms.Compose(
        [
            RandomHorizontalFlip(),
            ColorTransform(),
            ToTensor(),
            Normalization(),
        ]
    )


def get_finetune_train_transformation():
    return transforms.Compose(
        [
            ResizedRandomCrop(scale=(0.2, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalization(),
        ]
    )


def get_val_transformation():
    return transforms.Compose(
        [FixedResize(), ToTensor(), Normalization()]
    )
