import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
import torch
from torchvision.transforms.functional import InterpolationMode
import torchio as tio


class Standardizer(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['image'] = (sample['image'] - sample['image'].mean()) / max(sample['image'].std(), 1e-5)
        else:
            sample = (sample - sample.mean()) / max(sample.std(), 1e-5)
        return sample


class CopyChannel(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if isinstance(sample, dict):
            if sample['image'].size()[0] == 1:
                sample['image'] = sample['image'].repeat([3, 1, 1])
        else:
            if sample.size()[0] == 1:
                sample = sample.repeat([3, 1, 1])
        return sample

class Resize:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.resize_func = transforms.Resize(size, interpolation)

    def __call__(self, sample):
        if isinstance(sample, dict):
            transformed_sample = dict()
            for key, image in sample.items():
                transformed_sample[key] = self.resize_func(image)
        else:
            transformed_sample = self.resize_func(sample)    
        return transformed_sample

class ToNumpy(object):
    """
    Use this class to shut up "UserWarning: The given NumPy array is not writeable ..."
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return np.array(img)

class ToTensor:
    def __init__(self):
        self.operator = transforms.ToTensor()

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample["image"] = self.operator(sample["image"])
            sample["mask"] = self.operator(sample["mask"])
        else:
            samlple = self.operator(sample) 
        return sample

class ResizeKeepAspectRatio:
    def __init__(self, target_size, pad_val=0):
        self.pad_val = pad_val
        self.target_size = target_size

    def __call__(self, img):
        # step 1: decide the resize ratio = min(h_ratio, w_ratio)
        h, w = img.size()[-2:]
        target_h, target_w = self.target_size
        h_resize_ratio = target_h / h
        w_resize_ratio = target_w / w
        resize_ratio = np.minimum(h_resize_ratio, w_resize_ratio)

        # step 2: resize the img w.r.t the ratio
        resize_h = int(round(resize_ratio * h))
        resize_w = int(round(resize_ratio * w))
        resized_img = F.resize(img=img, size=(resize_h, resize_w))

        # pad the image
        # pad_left, pad_top, pad_right, pad_bottom
        pad_config = (0, 0, target_w-resize_w, target_h-resize_h)
        pad_img = F.pad(resized_img, pad_config, self.pad_val)
        return pad_img


class RandomResizedCrop:
    def __init__(self, p, scale, ratio, generator=None):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator

    def __call__(self, sample):
        if isinstance(sample, dict):
            i, j, h, w = transforms.RandomResizedCrop.get_params(sample["image"], self.scale, self.ratio)
            _, h_img, w_img = sample["image"].size()
            sample["image"] = F.crop(sample["image"], i, j, h, w)
            sample["image"] = F.resize(sample["image"], (h_img, w_img)) 
            sample["mask"] = F.crop(sample["mask"], i, j, h, w)
            sample["mask"] = F.resize(sample["mask"], (h_img, w_img)) 
             
        else:
            if self.generator.random() < self.p:
                i, j, h, w = transforms.RandomResizedCrop.get_params(sample, self.scale, self.ratio)
                _, h_img, w_img = sample.size()
                sample = F.crop(sample, i, j, h, w)
                sample = F.resize(sample, (h_img, w_img))
        return sample


class GaussianNoise(torch.nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.p=p
    
    def forward(self, sample):
        if isinstance(sample, dict):
            if torch.rand(1)<self.p:
                dtype = sample['image'].dtype
                if not sample['image'].is_floating_point():
                    sample['image'] = sample['image'].float()

                sigma = 25.0
                sample['image'] = sample['image'] + sigma * torch.randn_like(sample['image'])
                if sample['image'].dtype != dtype:
                    sample['image'] = sample['image'].to(dtype)
            
        else:
            if torch.rand(1)<self.p:
                dtype = sample.dtype
                if not sample.is_floating_point():
                    sample = sample.float()

                sigma = 25.0
                sample = sample + sigma * torch.randn_like(sample)
                if sample.dtype != dtype:
                    sample = sample.to(dtype)

        return sample


class RandomAffine(transforms.RandomAffine):
    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        if isinstance(sample, dict): 
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, sample["image"].size())
            
            sample["image"] = F.affine(sample["image"], *ret)
            sample["mask"] = F.affine(sample["mask"], *ret)
        else:
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, sample.size())
            sample = F.affine(sample, *ret)
            
        return sample

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            if isinstance(sample, dict): 
                sample["image"] = F.hflip(sample["image"])
                sample["mask"] = F.hflip(sample["mask"]) 
            else:
                sample = F.hflip(sample) 
            
        return sample

class RandomRotation(transforms.RandomRotation):
    def __call__(self, sample):
        angle = self.get_params(self.degrees)
        if isinstance(sample, dict): 
            sample["image"] = F.rotate(sample["image"], angle)
            sample["mask"] = F.rotate(sample["mask"], angle) 
        else:
            sample = F.rotate(sample, angle) 
            
        return sample

omni_augmentation = [
    transforms.RandomHorizontalFlip(p=0.5),
]
omni_tensor_transformation = [RandomResizedCrop(p=0.5, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                              transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3)),
                              transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6)),
                              transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8))]

standard_augmentation = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)),
]

ct_augmentation = [
    # transforms.RandomHorizontalFlip(p=0.25),
    RandomAffine(degrees=(-5, 5), translate=(0.1,0.1), scale=(0.8, 1.2)),
    transforms.RandomRotation(degrees=15)
]

# ct_strong_augmentation = [
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
#     transforms.RandomAffine(degrees=(-10, 10), translate=(0.2,0.2), scale=(0.8, 1.2)),
#     GaussianNoise(p=0.25),
#     transforms.RandomRotation(degrees=15)
# ]

ct_strong_augmentation = [
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(p=0.5, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
    RandomAffine(degrees=(-5, 5), translate=(0.1,0.1), scale=(0.8, 1.2)),
    GaussianNoise(p=0.15),
    RandomRotation(degrees=15)
]

weak_augmentation = [
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
]

ct_strong_tt_augmentation = [
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(p=0.5, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
    #RandomAffine(degrees=(-5, 5), translate=(0.1,0.1), scale=(0.8, 1.2)),
    # GaussianNoise(p=0.15),
    RandomRotation(degrees=10)
]

test_time_augmentation = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=(-15, 15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    ]

#to_tensor = [ToNumpy(), transforms.ToTensor(), Standardizer()]
to_tensor = [Standardizer()]


def compose_transform(augmentation=None, resize=None, image_format="greyscale"):
    """
    Function that composes the transformation workflow
    :param augmentation:
    :param resize:
    :return:
    """
    basic_transform = []
    
    # add augmentation
    if augmentation is not None:
        if augmentation == "standard":
            basic_transform += standard_augmentation
        elif augmentation == "omni":
            basic_transform += omni_augmentation
        elif augmentation == "weak":
            basic_transform += weak_augmentation
        elif augmentation == "test_time":
            basic_transform += test_time_augmentation
        elif augmentation == "ct":
            basic_transform += ct_augmentation
        elif augmentation == "ct_strong":
            basic_transform += ct_strong_augmentation
        elif augmentation == 'ct_strong_tt':
            basic_transform += ct_strong_tt_augmentation
        else:
            raise ValueError("invalid augmentation {}".format(augmentation))

    # old resize position
    if resize is not None:
        basic_transform += [Resize(resize)]
    
    # add to tensor and normalization
    basic_transform += to_tensor

    # resize position #1
    # if resize is not None:
    #     basic_transform += [ResizeKeepAspectRatio(target_size=resize)]

    # tensor level processing
    if augmentation is not None:
        if augmentation == "omni":
            basic_transform += omni_tensor_transformation

    # resize position #2
    # if resize is not None:
    #     basic_transform += [ResizeKeepAspectRatio(target_size=resize)]

    # add channel duplication
    if image_format == "greyscale":
        basic_transform += [CopyChannel()]
    return transforms.Compose(basic_transform)



class CopyChannel3D(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if isinstance(sample, dict):
            if sample['image'].size()[0] == 1:
                sample['image'] = sample['image'].repeat([3, 1, 1, 1])
        else:
            if sample.size()[0] == 1:
                sample = sample.repeat([3, 1, 1, 1])
        return sample


def compose_3D_transform(augmentation=None, resize=None, image_format="greyscale"):
    """
    Function that composes the transformation workflow
    :param augmentation:
    :param resize:
    :return:
    """
    basic_transform = []

    # add augmentation
    if augmentation is not None:
        if augmentation == "ct_strong":
            basic_transform += [
            tio.RandomFlip(axes=('LR',)),   # Random flip along Left-Right axis
            tio.RandomAffine(scales=(0.8, 1.2), degrees=(-5, 5), translation=(0.1,0.1)),  # Random affine transformations
            tio.RandomNoise(std=(0, 0.15)),  # Add random noise
            ]
        else:
            raise ValueError("invalid augmentation {}".format(augmentation))

    # old resize position
    if resize is not None:
        basic_transform += [tio.Resize(resize),]
    
    # add to tensor and normalization
    basic_transform += to_tensor

    # add channel duplication
    if image_format == "greyscale":
        basic_transform += [CopyChannel3D()]
    return tio.Compose(basic_transform)

