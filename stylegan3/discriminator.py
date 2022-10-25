"""
This file is not part of the stylegan3 repo. It is used for testing discriminator in the network.
"""
import os
import json
import pickle
from turtle import title
import torch
import torch.nn as nn
import torchvision
import numpy as np
#import torchvision.io as io
#import torchvision.transforms as transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import statsmodels.api as sm
import pylab as py
import kornia
import cv2

with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
    D = pickle.load(f)['D'].cuda()  # torch.nn.Module
resize = torchvision.transforms.Resize((1024,1024))

def main():
    #convert_rbg_to_lab("ffhq_images_1024x1024/*","ffhq_images_1024x1024_lab_format/*")
    files = glob.glob("CleanedAsianMenLongHair/*")
    result = []
    for file in files:
        pic = torchvision.io.read_image(file).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        result.append(float(score.cpu().numpy()[0][0])) 
    print(result)
    """
    files = ["dictionary9.txt","dictionary19.txt","dictionary29.txt","dictionary39.txt","dictionary49.txt"]
    dict_list = []
    for file in files:
        with open(file, 'r') as convert_file:
            dictionary = convert_file.read()
            dict_list.append(json.loads(dictionary))
    
    complete_dict = {}
    for one_dict in dict_list:
        complete_dict = complete_dict | one_dict
    test = np.array(list(complete_dict.values()))
    print(test.shape)
    sm.qqplot(test, line='45')
    plt.show()
    complete_dict = dict(sorted(complete_dict.items(), key=lambda item: item[1]))

    scores = list(complete_dict.values())
    npscores = np.array(scores)
    print(f"Mean of scores is:{np.mean(npscores)}    Std of scores is:{np.std(npscores)} Median of scores is:{np.median(npscores)}")
    x = list(range(50000))
    plt.figure(figsize=(15,10))
    plt.title("Scores for all 1024x1024 images")
    plt.ylabel("score")
    sm.qqplot(npscores, line ='45')
    py.show()
    plt.hist(scores, bins=1000)
    plt.savefig("score.png")

    files = list(complete_dict.keys())
    show_images(files[0:100],"Least 100")
    show_images(files[49900:], "Top 100")
    """
    

def show_images(files, figurename):
    images = []
    for file in files:
        name = glob.glob("ffhq_images_1024x1024/*/"+str(file))
        #print(name)
        image = Image.open(name[0])
        images.append(image)

        
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.savefig(figurename+".png")


def read_images_get_scores():
    """
    This function reads all the images and 
    """
    folders = os.listdir("ffhq_images_1024x1024")
    folders.sort()
    for i in range(9,50,10):
        images = {}
        group = folders[i-9:i+1]
        txt = "dictionary"+str(i)+".txt"
        for folder in group:
            files = os.listdir("ffhq_images_1024x1024/"+folder) 
            print(f"Reading images from directory: ffhq_images_1024x1024/{folder}")
            for file in files:
                pic = torchvision.io.read_image("ffhq_images_1024x1024/"+folder+"/"+file).cuda()
                score = D(pic[None,:,:,:],c=None).cuda()
                images[file] = float(score.cpu().numpy()[0][0])
        with open(txt, 'w') as convert_file:
            convert_file.write(json.dumps(images))


def convert_rbg_to_lab(read_dir, write_dir):
    """
    This is a mutator that converts all images in ffhq_images_1024x1024 folders 
        from RBG image format to LAB image format. Link to explanation of LAB 
        image format: https://en.wikipedia.org/wiki/CIELAB_color_space
    Args:
        read_dir (string): the directory for reading the images
        write_dir (string): the directory for writing the converted images
    """    
    read_folders = glob.glob(read_dir, recursive=False)
    read_folders.sort()
    write_folders = glob.glob(write_dir,recursive=False)
    write_folders.sort()
    read_folders = list(read_folders)
    write_folders = list(write_folders)

    for i,(read_folder, write_folder) in enumerate(zip(read_folders, write_folders)): 
        read_folder = read_folder + "/*"
        image_paths = glob.glob(read_folder)
        for one_path in image_paths:
            print(f"Reading images from folder {one_path} and writing to folder {write_folder}")
            img = cv2.imread(one_path)
            splits = one_path.split("/")
            file_name = write_folder+"/"+splits[2]
            LAB_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            print(file_name)
            cv2.imwrite(file_name,LAB_image)
            break
        break

# bad generated images in out directory: 5195, 5196, 5197, 5198, 5200, 5201, 


if __name__=="__main__":
    main()