"""
This file is not part of the stylegan3 repo. It is used for testing discriminator in the network.
"""
import os
import os.path
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
import sys

gan_model_filename = "/mnt/data/students/models/fairface_stylegan3/training-runs/00011-stylegan3-t-train_prepared-gpus8-batch32-gamma8.2/network-snapshot-007000.pkl"
#with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
with open(gan_model_filename, 'rb') as f:
    D = pickle.load(f)['D'].cuda()  # torch.nn.Module
resize = torchvision.transforms.Resize((1024,1024))

def main():
    images_dir = "/mnt/data/students/fairface/data/padding_0.25/train_prepared"
    xyz_outdir = images_dir + "_lab_format"
    print(f"Converting {images_dir} to xyz and outputting to {xyz_outdir}")
    duplicate_subdirectories(images_dir, xyz_outdir)
    #convert_rbg_to_ciexyz(images_dir + "/*", xyz_outdir + "/*")
    #read_images_get_scores(xyz_outdir)

    ### below not yet fixed
    get_luminance(xyz_outdir, in_csv_filename, out_csv_filename)
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
   
def duplicate_subdirectories(root_path, target_path):
    dirs = []
    rootdir = root_path
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            #print(d)
            dirs.append(d)
    for dir in dirs:
        target_dir = target_path + "/" + os.path.basename(dir)
        if not os.path.exists(target_dir):
            print(f"Creating {target_dir}")
            os.mkdir(target_dir)
    
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

def get_correlation(files, complete_dict, images_dir, out_csv_filename):
    lab = []
    scores = []
    index = 0
    for file in files:
        #name = glob.glob("ffhq_images_1024x1024/*/"+str(file))
        name = glob.glob(images_dir + "/*/" + str(file))
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
    #df.to_csv("RGB_format_images_data.csv)
    df.to_csv(out_csv_filename)
    return np.corrcoef(np.array(scores), lab)

def get_luminance(images_dir, in_csv_filename, out_csv_filename):
    df = pd.read_csv(in_csv_filename)
    image_ids = df["image_id"]
    lab = []
    for index, id in enumerate(image_ids):
        #file = glob.glob("ffhq_images_1024x1024/*/"+id)
        file = glob.glob(images_dir + "/*/"+id)
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
    df.to_csv(out_csv_filename)
        
    
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


def read_images_get_scores(images_dir):
    """
    This function reads all the images and 
    """
    #folders = os.listdir("ffhq_images_1024x1024_lab_format")
    folders = os.listdir(images_dir)
    folders.sort()
    for folder in folders:
        images = {}
        #group = folders[i-9:i+1]
        #txt = "dictionary"+str(i)+".txt"
        txt = "dictionary" + folder + ".txt"
        #for folder in group:
        files = os.listdir(images_dir + "/" + folder) 
        print(f"Scoring images from directory: {images_dir}/{folder}")
        for file in files:
            #pic = torchvision.io.read_image("ffhq_images_1024x1024_lab_format/"+folder+"/"+file).cuda()
            pic = torchvision.io.read_image(images_dir+ "/" + folder+"/"+file).cuda()
            score = D(pic[None,:,:,:],c=None).cuda()
            images[file] = float(score.cpu().numpy()[0][0])
    with open(txt, 'w') as convert_file:
        convert_file.write(json.dumps(images))

def get_average_color(images_dir, csv_filename):
    #filename = "correct_LAB_format_images_data.csv"
    filename = csv_filename
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
        file = glob.glob(images_dir + "/*/"+id)[0]
        image = torchvision.io.read_image(file)
        image  = image.float()
        red_mean, green_mean, blue_mean = torch.mean(image,dim=[1,2]).numpy()
        r.append(red_mean)
        g.append(green_mean)
        b.append(blue_mean)
    df["red_mean"] = r
    df["green_mean"] = g
    df["blue_mean"] = b
    #df.to_csv("correct_LAB_format_images_data.csv")
    df.to_csv(csv_filename)

def get_average_color(out_filename, in_dirname):
    filename = out_filename
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
        #file = glob.glob("ffhq_images_1024x1024/*/"+id)[0]
        file = glob.glob(in_dirname + "/*/" + id)[0]
        image = torchvision.io.read_image(file)
        image  = image.float()
        red_mean, green_mean, blue_mean = torch.mean(image,dim=[1,2]).numpy()
        r.append(red_mean)
        g.append(green_mean)
        b.append(blue_mean)
    df["red_mean"] = r
    df["green_mean"] = g
    df["blue_mean"] = b
    df.to_csv(filename)


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
        #print(read_folder)
        read_folder = read_folder + "/*"
        image_paths = glob.glob(read_folder)
        print(f"Reading images from folder {read_folder} and writing to folder {write_folder}")
        for one_path in image_paths:
            #print("one_path: ",one_path)
            img = cv2.imread(one_path)
            img = img.astype("float32")
            img = img/255.0
            #splits = one_path.split("/")
            #file_name = write_folder+"/"+splits[2]
            file_name = write_folder + "/" + os.path.basename(one_path)
            LAB_image = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)           
            cv2.imwrite(file_name,LAB_image)
        print("Done.")

def get_data_botton100_top100():
    top = glob.glob("top_least_100_images/top_100/rgb/*")
    bottom = glob.glob("top_least_100_images/least_100/rgb/*")
    df = pd.read_csv("correct_LAB_format_images_data.csv",index_col=False)
    new_df = pd.DataFrame()
    image_ids = []
    scores = []
    rmean = []
    gmean = []
    bmean = []
    luminance = []
    for bottom_image, top_image in zip(bottom,top):
        names1 = bottom_image.split("/")
        name1 = names1[len(names1)-1].strip()
        names2 = top_image.split("/")
        name2 = names2[len(names2)-1].strip()
        
        new_df = df.loc[(df["image_id"]==name1)|(df["image_id"]==name2)][["image_id","scores_rgb","red_mean","green_mean","blue_mean","luminance"]]
        series1 = new_df.iloc[0]
        series2 = new_df.iloc[1]
        
        image_ids.append(series1["image_id"])
        image_ids.append(series2["image_id"])
        
        scores.append(series1["scores_rgb"])
        scores.append(series2["scores_rgb"])
        
        rmean.append(series1["red_mean"])
        rmean.append(series2["red_mean"])
        
        gmean.append(series1["green_mean"])
        gmean.append(series2["green_mean"])
        
        bmean.append(series1["blue_mean"])
        bmean.append(series2["blue_mean"])
        
        luminance.append(series1["luminance"])
        luminance.append(series2["luminance"])
    dictionary = {"discriminator_score":scores,"image":image_ids,"red_mean":rmean,"green_mean":gmean,"blue_mean":bmean,"luminance":luminance}
    result = pd.DataFrame(dictionary)
    result.to_csv("bottom_top_100_data.csv",index=False)

if __name__=="__main__":
    main()
