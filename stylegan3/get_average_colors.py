import os
import os.path
import pickle
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import pylab as py
import cv2
import click
import sys
import argparse
import time
from torchvision.transforms.functional import get_image_size

def get_discriminator(gan_model_filename):
    with open(gan_model_filename, 'rb') as f:
        D = pickle.load(f)['D'].cuda()  # torch.nn.Module
    return D

def get_average_colors(image_filename):
    file = image_filename
    image = torchvision.io.read_image(file)
    image  = image.float()
    red_mean, green_mean, blue_mean = torch.mean(image,dim=[1,2]).numpy()
    return red_mean, green_mean, blue_mean


def get_score(image_filename, discriminator, model_image_size, antialias=False):
    D = discriminator
    pic = torchvision.io.read_image(image_filename).cuda()
    image_size = get_image_size(pic)
    if image_size != model_image_size or antialias == True:       
        pic = torchvision.transforms.functional.resize(pic, model_image_size, antialias=antialias)
        
        
    score = D(pic[None,:,:,:],c=None).cuda()
    score = score.item()
    return score

def get_luminance(image_filename):
    image = cv2.imread(image_filename)
    image = image.astype("float32")
    image = image/255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    luminance = image[:,:,1]
    mean = np.mean(np.array(luminance))
    return mean


def process_images(images_dir, discriminator, out_csv_filename, model_image_size):
    D = discriminator
    folders = os.listdir(images_dir)
    folders = [f for f in folders if os.path.isdir(images_dir + "/" + f)]
    folders.sort()
    images = {}
    for folder in folders:
        files = os.listdir(images_dir + "/" + folder)
        files = [f for f in files if (f.endswith('.png') or f.endswith('.jpg'))]
        print(f"Processing images from directory: {images_dir}/{folder}")
        for file in files:
            images[file] = {}
            image_absolute_path = images_dir + "/" + folder + "/" + file
            pic = torchvision.io.read_image(image_absolute_path).cuda()
            score = get_score(image_absolute_path, D, model_image_size)
            images[file]["scores_rgb"] = score
            r, g, b = get_average_colors(image_absolute_path)
            images[file]["red_mean"] = r
            images[file]["green_mean"] = g
            images[file]["blue_mean"] = b
            images[file]["luminance"] = get_luminance(image_absolute_path)
    df = pd.DataFrame(images)
    print(f"Writing to {out_csv_filename}")
    df = df.transpose()
    df.to_csv(out_csv_filename)
    print("Done.")
    print(df.head())
          
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Generate CSV file with RGB and luminance values.')
    parser.add_argument('--images_dir', type=str,help="root directory of processed images")
    parser.add_argument('--dest_csv', type=str, help="full path and filename of the destination CSV file.")
    parser.add_argument('--gan_model', type=str, help="file name of GAN model to use.  Assumes last part of name (before .pkl) is the width and height of the images in the modal, e.g., my_model256x256.pkl.")
    #parser.add_argument('--resize', type=int, nargs=1, required=False, default=None,help="Optionally resize images to this height while maintaining aspect ratio..  For example: --resize 1024")
    
    args = parser.parse_args()
    images_dir = args.images_dir
    discriminator = get_discriminator(args.gan_model)
    out_csv_filename = args.dest_csv
    #resize = args.resize
    ### find dimensions use by model for resizing
    model_prefix = os.path.splitext(args.gan_model)[0]
    model_image_height = int(model_prefix.split('x')[1])
    dimensions = (model_image_height, model_image_height)
    
    process_images(images_dir, discriminator, out_csv_filename, model_image_height)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"Took {elapsed_time} minutes.")

if __name__ == "__main__":
    main()
