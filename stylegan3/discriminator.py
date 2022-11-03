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
import pandas as pd
#import torchvision.io as io
#import torchvision.transforms as transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import statsmodels.api as sm
import pylab as py
import cv2
import scipy.stats as stats

with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
    D = pickle.load(f)['D'].cuda()  # torch.nn.Module
resize = torchvision.transforms.Resize((1024,1024))

def main():
    #convert_rbg_to_lab("ffhq_images_1024x1024/*","ffhq_images_1024x1024_lab_format/*")
    #read_images_get_scores()
    dict_files = glob.glob("parsing_dictionaries/lab/*") 
    print(dict_files)
    dict_list = []
    for file in dict_files:
        with open(file, 'r') as convert_file:
            dictionary = convert_file.read()
            dict_list.append(json.loads(dictionary))
    
    complete_dict = {}
    for one_dict in dict_list:
        complete_dict = complete_dict | one_dict
    complete_dict = dict(sorted(complete_dict.items(), key=lambda item: item[1]))

    files = list(complete_dict.keys())
    least_100_files = files[0:100]
    top_100_files = files[69900:]
    scores = list(complete_dict.values())
    npscores = np.array(scores)
    plot_normal_quantile(npscores)
    np.savetxt("score_for_lab_images.csv",npscores,delimiter=",")
    print(f"Mean of scores is:{np.mean(npscores)}    \
        Std of scores is:{np.std(npscores)} Median of scores is:{np.median(npscores)}")
    print("Correlation for all image is:", get_correlation(files, complete_dict))

    plt.figure(figsize=(15,10))
    plt.title("Scores for all 1024x1024 lab images")
    plt.ylabel("score")
    plt.hist(scores, bins=1000)
    plt.savefig("score.png")

    """ files = list(complete_dict.keys())
    show_images(files[0:100],"Least 100")
    show_images(files[69900:], "Top 100") """
   
    
def make_subdirectories(root_path):
    paths = ['00000', '01000', '02000', '03000', '04000', '05000', '06000', '07000', '08000', '09000', '10000', '11000', 
             '12000', '13000', '14000', '15000', '16000', '17000', '18000', '19000', '20000', '21000', '22000', '23000', 
             '24000', '25000', '26000', '27000', '28000', '29000', '30000', '31000', '32000', '33000', '34000', '35000', 
             '36000', '37000', '38000', '39000', '40000', '41000', '42000', '43000', '44000', '45000', '46000', '47000', 
             '48000', '49000', '50000', '51000', '52000', '53000', '54000', '55000', '56000', '57000', '58000', '59000', 
             '60000', '61000', '62000', '63000', '64000', '65000', '66000', '67000', '68000', '69000']
    for items in paths:
        path = os.path.join(root_path, items)
        os.makedirs(path)
    
def get_correlation(files, complete_dict):
    lab = []
    scores = []
    index = 0
    for file in files:
        name = glob.glob("ffhq_images_1024x1024_lab_format/*/"+str(file))
        score = complete_dict[file]
        scores.append(score)
        image = cv2.imread(name[0])
        luminance = image[:,:,0]
        mean = np.mean(np.array(luminance))
        lab.append(mean)
        if index % 100 ==0:
            print(f"working on image {file}")
        index +=1

    dic_df = {"image_id":files,"scores":scores,"luminance":lab}
    df = pd.DataFrame(dic_df)
    df.to_csv("LAB_format_images_data.csv")
    return np.corrcoef(np.array(scores), lab)



def plot_normal_quantile(input):
    sm.qqplot(input, line="45")
    plt.savefig("images/normal_qq_plot.png")

def show_images(files, figurename):
    images = []
    for file in files:
        name = glob.glob("ffhq_images_1024x1024_lab_format/*/"+str(file))
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
    folders = os.listdir("ffhq_images_1024x1024_lab_format")
    folders.sort()
    for i in range(9,70,10):
        images = {}
        group = folders[i-9:i+1]
        txt = "dictionary"+str(i)+".txt"
        for folder in group:
            files = os.listdir("ffhq_images_1024x1024_lab_format/"+folder) 
            print(f"Reading images from directory: ffhq_images_1024x1024_lab_format/{folder}")
            for file in files:
                pic = torchvision.io.read_image("ffhq_images_1024x1024_lab_format/"+folder+"/"+file).cuda()
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
        print(f"Reading images from folder {read_folder} and writing to folder {write_folder}")
        for one_path in image_paths:
            img = cv2.imread(one_path)
            splits = one_path.split("/")
            file_name = write_folder+"/"+splits[2]
            LAB_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            cv2.imwrite(file_name,LAB_image)

# bad generated images in out directory: 5195, 5196, 5197, 5198, 5200, 5201, 


if __name__=="__main__":
    main()