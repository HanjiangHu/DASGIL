from torchvision import transforms
import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import copy
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_subdirectory(dir):
    subdirectory = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for _, dir, _ in os.walk(dir):
        if dir != []:
            subdirectory = dir
            break
    return subdirectory


class util():
    def mkdirs(self, paths):
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                self.mkdir(path)
        else:
            self.mkdir(paths)

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


def write_loss_image(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)

    images_seg = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('output' in attr or 'input' in attr) and ('seg' in attr)]
    images_others = [attr for attr in dir(trainer) \
                  if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                              'output' in attr or 'input' in attr) and ('seg' not in attr)]
    for img in images_seg:
        train_writer.add_images(img, tensor2array_rainbow(getattr(trainer, img)[0], colormap='rainbow',seg=True),
                                iterations + 1, dataformats='CHW')

    for img in images_others:
        train_writer.add_images(img, tensor2array_rainbow(getattr(trainer, img)[0], colormap='rainbow'), iterations + 1, dataformats='CHW')


# visualize the depth map
def opencv_rainbow(resolution=1000):
    opencv_rainbow_data = (
        (0.000, (0.00, 1.00, 1.00)),
        (0.001, (0.00, 0.00, 1.00)),
        (0.100, (1.00, 0.00, 1.00)),
        (0.300, (1.00, 0.00, 0.00)),
        (0.500, (1.00, 1.00, 0.00)),
        (0.700, (0.00, 1.00, 0.00)),
        (1.000, (0.00, 1.00, 1.00))
    )
    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow()}


def tensor2array_rainbow(tensor, max_value=None, colormap='rainbow',seg=False,test_save=False):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = (COLORMAPS[colormap](norm_array)*255).astype(np.uint8)#

    elif tensor.ndimension() == 3:
        if not seg:
            assert(tensor.size(0) == 3)
            array = 0.5 + tensor.numpy()*0.5
        else:
            normalize = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            tensor = visualize_seg(tensor) # from tensor class mask to torch tensor rgb
            tensor = normalize(tensor)
            if test_save:
                tensor = tensor.transpose(0, 2).transpose(0, 1)
            array = 0.5 + tensor.numpy() * 0.5
    return array


# visualize the segmentation map
def visualize_seg(mask):
    mask_np = mask.numpy()
    rgb_np = copy.deepcopy(mask_np)
    index2color = [(0, 0, 0), (210, 0, 200),(90, 200,255),(0, 199, 0),(90, 240, 0),(140,140,140),(100,60,100),(250,100,255),(255,255, 0),
                   (200,200,  0),(255,130, 0),(80,80,80),(160,60,60),(255, 127, 80),(0,  139,  139)]
    for h in range(mask.size(1)):
        for w in range(mask.size(2)):
            rgb_np[0][h][w], rgb_np[1][h][w], rgb_np[2][h][w] = index2color[int(mask_np[0][h][w])]
    return torch.from_numpy(rgb_np).float()

