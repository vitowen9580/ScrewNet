from __future__ import print_function 
import argparse 
from argparse import Namespace 
from tensorflow.compat.v1 import ConfigProto 
from tensorflow.compat.v1 import InteractiveSession
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_gpu(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id # Set the GPU to use
    config = ConfigProto()
    config.gpu_options.allow_growth = True # Dynamically grow the memory
    session = InteractiveSession(config=config) # Start the session


            
def set_img_color(img, predict_mask, weight_foreground, grayscale):
    # if grayscale:
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    origin = img
    img[np.where(predict_mask == 0)] = (0,0,255)
    cv2.addWeighted(img, weight_foreground, origin, (1 - weight_foreground), 0, img)
    return img

def plot_training_history(history, path=None):
    """
    Plot training history (loss and accuracy).

    Parameters:
    - history: History object from model.fit(), or a dictionary containing training history.
    - path: Optional path to save the plots.
    """
    # Extract history dictionary from History object or assume it's already a dictionary
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(history_dict['loss'], label='Training Loss')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plot training accuracy
    plt.figure(figsize=(10, 5))
    if 'accuracy' in history_dict:
        plt.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict:
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Save the plot if a path is provided
    if path:
        plt.savefig(path)
    else:
        plt.show()

def add_pepper_noise(images, amount=0.01):

    noisy_images = np.copy(images)
    num_images, height, width, channels = noisy_images.shape
    num_pepper_pixels = int(amount * height * width)

    # Generate random coordinates for pepper noise for all images
    pepper_coords = [np.random.randint(0, i, (num_images, num_pepper_pixels)) for i in [height, width]]

    # Add pepper noise to the images
    noisy_images[np.arange(num_images)[:, None], pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_images



import copy
import os
import shutil

import numpy as np
import torch

# To make directories 
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

# For Pytorch data loader
def create_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['valA'] = os.path.join(dataset_dir, 'lvalA')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], 'Link'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], 'Link'))

    return dirs



def get_traindata_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    return dirs

def get_valdata_link(dataset_dir):
    dirs = {}
    dirs['valA'] = os.path.join(dataset_dir, 'lvalA')
    return dirs

# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')