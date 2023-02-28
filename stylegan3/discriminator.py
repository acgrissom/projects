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
from sklearn.linear_model import LinearRegression
import click


with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
    D = pickle.load(f)['D'].cuda()  # torch.nn.Module
resize = torchvision.transforms.Resize((1024,1024))

def main():
    #convert_rbg_to_ciexyz("ffhq_images_1024x1024/*","ffhq_images_1024x1024_lab_format/*")
    #read_images_get_scores()
    
    #get_luminance()

    
    """ clf = LinearRegression()
    df = pd.read_csv("correct_LAB_format_images_data.csv")
    luminance = df["luminance"].to_numpy()
    luminance_reshaped = np.array(np.split(luminance, len(luminance)))
    scores = df["scores_rgb"].to_numpy()
    scores_reshaped = np.array(np.split(scores, len(scores)))
    
    clf.fit(luminance_reshaped,scores_reshaped)
    predictions = clf.predict(luminance_reshaped)
    predictions = np.squeeze(predictions)
    plt.figure(figsize=(10,10))
    plt.title("scores for images vs luminance")
    plt.xlabel("Luminance")
    plt.ylabel("Scores")
    plt.scatter(luminance, scores)
    plt.plot(luminance, predictions, c="red",lw=4.0)
    plt.savefig("images/score_luminance.png") """
    
    """ dict_files = glob.glob("parsing_dictionaries/rgb/*") 
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
    print(f"Mean of scores is:{np.mean(npscores)}    \
        Std of scores is:{np.std(npscores)} Median of scores is:{np.median(npscores)}")
    print("Correlation for all image is:", get_correlation(files, complete_dict)) """

    """     
    plt.figure(figsize=(15,10))
    plt.title("Scores for all 1024x1024 lab images")
    plt.ylabel("score")
    plt.hist(scores, bins=1000)
    plt.savefig("score.png")
     """

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

def visualize_score(format):
    format = format.upper()
    file = format+"_format_images_data.csv"
    df = pd.read_csv(file)
    scores = df["scores"].to_numpy()
    luminance = df["luminance"].to_numpy()
    plt.figure(figsize=(10,10))
    plt.title("scores for "+format+" images vs luminance")
    plt.xlabel("Luminance")
    plt.ylabel("Scores")
    plt.plot(scores, luminance)
    plt.savefig("images/"+format+".jpg")

def get_correct_df():
    df1 = pd.read_csv("LAB_format_images_data.csv")
    df2 = pd.read_csv("RGB_format_images_data.csv")
    new = pd.merge(df1,df2,how="left",on=["image_id"], indicator=True)
    new.drop(columns=["Unnamed: 0_x","Unnamed: 0_y","_merge","luminance_y"], inplace=True)
    new["scores_rgb"] = new["scores_y"]
    new.drop(columns=["scores_x", "scores_y"], inplace=True)
    new.rename(columns={"luminance_x":"luminance"}, inplace=True)
    print(new.head())
    new.to_csv("correct_LAB_format_images_data.csv")

def get_correlation(files, complete_dict):
    lab = []
    scores = []
    index = 0
    for file in files:
        name = glob.glob("ffhq_images_1024x1024/*/"+str(file))
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
    df.to_csv("RGB_format_images_data.csv")
    return np.corrcoef(np.array(scores), lab)

def get_luminance():
    df = pd.read_csv("correct_LAB_format_images_data.csv")
    image_ids = df["image_id"]
    lab = []
    for index, id in enumerate(image_ids):
        file = glob.glob("ffhq_images_1024x1024/*/"+id)
        image = cv2.imread(file[0])
        image = image.astype("float32")
        image = image/255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        luminance = image[:,:,1]
        mean = np.mean(np.array(luminance))
        lab.append(mean)
        if index % 100 ==0:
            print(f"working on image {file}")
    dictionary = {"luminance":lab}
    temp = pd.DataFrame(dictionary)
    temp.to_csv("luminance.csv")
    df["normalized_luminance"]=lab
    df.to_csv("correct_LAB_format_images_data.csv")
        
    
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

def get_average_color():
    filename = "correct_LAB_format_images_data.csv"
    try:
        with open("bottle.py") as f:
            print(f"Found {filename} in current directory")
    except FileNotFoundError:
        print(f'{filename} is not present')
    df = pd.read_csv(filename)
    image_ids = df["image_id"]
    r = []
    g = []
    b = []
    for id in image_ids:
        file = glob.glob("ffhq_images_1024x1024/*/"+id)[0]
        image = torchvision.io.read_image(file)
        image  = image.float()
        red_mean, green_mean, blue_mean = torch.mean(image,dim=[1,2]).numpy()
        r.append(red_mean)
        g.append(green_mean)
        b.append(blue_mean)
    df["red_mean"] = r
    df["green_mean"] = g
    df["blue_mean"] = b
    df.to_csv("correct_LAB_format_images_data.csv")

def convert_rbg_to_ciexyz(read_dir, write_dir):
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
            img = img.astype("float32")
            img = img/255.0
            splits = one_path.split("/")
            file_name = write_folder+"/"+splits[2]
            LAB_image = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
            cv2.imwrite(file_name,LAB_image)

# bad generated images in out directory: 5195, 5196, 5197, 5198, 5200, 5201, 


if __name__=="__main__":
    main()