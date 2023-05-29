import os
import os.path
#import json
import pickle
#from turtle import title
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
#import torchvision.io as io
#import torchvision.transforms as transforms
from PIL import Image
#import glob
#import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#import statsmodels.api as sm
import pylab as py
import cv2
#import scipy.stats as stats
#from sklearn.linear_model import LinearRegression
import click
import sys
import argparse

gan_model_filename = "/mnt/data/students/models/fairface_stylegan3/training-runs/00011-stylegan3-t-train_prepared-gpus8-batch32-gamma8.2/network-snapshot-007000.pkl"

def get_discriminator(gan_model_filename):
    with open(gan_model_filename, 'rb') as f:
        D = pickle.load(f)['D'].cuda()  # torch.nn.Module
    resize = torchvision.transforms.Resize((1024,1024))
    return D

def get_average_colors(image_filename):
    file = image_filename
    image = torchvision.io.read_image(file)
    image  = image.float()
    red_mean, green_mean, blue_mean = torch.mean(image,dim=[1,2]).numpy()
    return red_mean, green_mean, blue_mean


def get_score(image_filename, discriminator):
    D = discriminator
    pic = torchvision.io.read_image(image_filename).cuda()
    return D(pic[None,:,:,:],c=None).cuda()

def get_luminance(image_filename):
    image = cv2.imread(image_filename)
    image = image.astype("float32")
    image = image/255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    luminance = image[:,:,1]
    mean = np.mean(np.array(luminance))
    return mean


def process_images(images_dir, discriminator, out_csv_filename):
    D = discriminator
    folders = os.listdir(images_dir)
    folders = [f for f in folders if os.path.isdir(images_dir + "/" + f)]
    print(folders)
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
            score = get_score(image_absolute_path, D)
            images[file]["score_rgb"] = float(score.cpu().numpy()[0][0])
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
    parser = argparse.ArgumentParser(description='Generate CSV file with RGB and luminance values.')
    parser.add_argument('--images_dir', type=str,help="root directory of processed images")
    parser.add_argument('--dest_csv', type=str, help="full path and filename of the destination CSV file.")
    args = parser.parse_args()
    #images_dir = "/mnt/data/students/fairface/data/padding_0.25/train_prepared"
    images_dir = args.images_dir
    discriminator = get_discriminator(gan_model_filename)
    #out_csv_filename = "/mnt/data/students/fairface/data/fairface_data.csv"
    out_csv_filename = args.dest_csv
    process_images(images_dir, discriminator, out_csv_filename)
    pass

if __name__ == "__main__":
    main()
