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
import urllib.request

with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
    D = pickle.load(f)['D'].cuda()  # torch.nn.Module
resize = torchvision.transforms.Resize((1024,1024))

# with urllib.request.urlopen('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl') as f:
#     print("before")
#     D = pickle.load(f)['D'].cuda()  # torch.nn.Module
#     print("after")
# resize = torchvision.transforms.Resize((1024,1024))


def main_atypical():
    # df = pd.DataFrame(columns=['image', 'race', 'hair_length', 'luminance', 'rgb_score', 'red_mean', 'green_mean', 'blue_mean'])

    asian_files = glob.glob("images/AtypicalFaces/rbg_images/CleanedAsianMenLongHair/*")
    asian_results = np.empty((100,))
    asian_dict = {}

    black_files = glob.glob("images/AtypicalFaces/rbg_images/CleanedBlackMenLongHair/*")
    black_results = np.empty((100,))
    black_dict = {}

    white_files = glob.glob("images/AtypicalFaces/rbg_images/CleanedWhiteMenLongHair/*")
    white_results = np.empty((100,))
    white_dict = {}

    for i, file in enumerate(asian_files):
        pic = torchvision.io.read_image(file).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        asian_results[i] = score.item()
        if (score.item() in asian_dict.keys()):
            print(asian_dict[score.item()])
            print(file,"\n")
        asian_dict[score.item()] = file

    for i, file in enumerate(black_files):
        pic = torchvision.io.read_image(file).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        black_results[i] = score.item()
        if (score.item() in black_dict.keys()):
            print(black_dict[score.item()])
            print(file,"\n")
        black_dict[score.item()] = file

    for i, file in enumerate(white_files):
        pic = torchvision.io.read_image(file).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        white_results[i] = score.item()
        if (score.item() in white_dict.keys()):
            print(white_dict[score.item()])
            print(file,"\n")
        white_dict[score.item()] = file

    print("Asian Men Long Hair Mean:", asian_results.mean())
    print("Black Men Long Hair Mean:", black_results.mean())
    print("White Men Long Hair Mean:", white_results.mean())

    print("Asian Men Long Hair StdDev:", asian_results.std())
    print("Black Men Long Hair StdDev:", black_results.std())
    print("White Men Long Hair StdDev:", white_results.std())

    # plot all scores of images
    asian_results_sorted = np.sort(asian_results)
    black_results_sorted = np.sort(black_results)
    white_results_sorted = np.sort(white_results)

    x = np.linspace(1,100,100)

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, asian_results_sorted, label="Asian")
    plt.plot(x, black_results_sorted, label="Black")
    plt.plot(x, white_results_sorted, label="White")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/AtypicalFacesScores.png")
    plt.clf()

    # plot lowest scoring images
    lowest_images = []
    for i in range(10):
        lowest_images.append(asian_dict[asian_results_sorted[i]])
    for i in range(10):
        lowest_images.append(black_dict[black_results_sorted[i]])
    for i in range(10):
        lowest_images.append(white_dict[white_results_sorted[i]])

    show_images(lowest_images, "images/AtypicalFaces/plots/LowestScoresAtypical", "10 Lowest Scoring Atypical Faces by Race")
    plt.clf()

    # plot highest scoring images
    highest_images = []
    for i in range(99,89,-1):
        highest_images.append(asian_dict[asian_results_sorted[i]])
    for i in range(99,89,-1):
        highest_images.append(black_dict[black_results_sorted[i]])
    for i in range(99,89,-1):
        highest_images.append(white_dict[white_results_sorted[i]])

    show_images(highest_images, "images/AtypicalFaces/plots/HighestScoresAtypical", "10 Highest Scoring Atypical Faces by Race")
    plt.clf()

    # create dataframe with image, score, race, hair, average rgb
    asian_df = pd.DataFrame.from_dict(asian_dict, orient='index').reset_index()
    asian_df = asian_df.rename(columns={0: 'image', 'index':'discriminator_score'})
    asian_df['race'] = 'asian'
    asian_df['hair_length'] = 'long'

    black_df = pd.DataFrame.from_dict(black_dict, orient='index').reset_index()
    black_df = black_df.rename(columns={0: 'image', 'index':'discriminator_score'})
    black_df['race'] = 'black'
    black_df['hair_length'] = 'long'

    white_df = pd.DataFrame.from_dict(white_dict, orient='index').reset_index()
    white_df = white_df.rename(columns={0: 'image', 'index':'discriminator_score'})
    white_df['race'] = 'white'
    white_df['hair_length'] = 'long'

    df = pd.concat([asian_df, black_df, white_df])

    r = []
    g = []
    b = []

    for i, row in df.iterrows():
        red_mean, green_mean, blue_mean = get_average_color_image(row['image'])
        r.append(red_mean)
        g.append(green_mean)
        b.append(blue_mean)

    df["red_mean"] = r
    df["green_mean"] = g
    df["blue_mean"] = b

    lab = []
    for i, row in df.iterrows():
        path = row['image']
        path = path.replace("rbg_images", "lab_images")
        image = cv2.imread(path)
        luminance = image[:,:,0]
        mean = np.mean(np.array(luminance))
        lab.append(mean)
    df["luminance"] = lab

    return asian_results_sorted, black_results_sorted, white_results_sorted, df



def main_typical():
    asian_files = glob.glob("images/AtypicalFaces/rbg_images/CleanedAsianMenShortHair/*")
    asian_results = np.empty((100,))
    asian_dict = {}

    black_files = glob.glob("images/AtypicalFaces/rbg_images/CleanedBlackMenShortHair/*")
    black_results = np.empty((100,))
    black_dict = {}

    white_files = glob.glob("images/AtypicalFaces/rbg_images/CleanedWhiteMenShortHair/*")
    white_results = np.empty((100,))
    white_dict = {}

    for i, file in enumerate(asian_files):
        pic = torchvision.io.read_image(file).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        asian_results[i] = score.item()
        if (score.item() in asian_dict.keys()):
            print(asian_dict[score.item()])
            print(file,"\n")
        asian_dict[score.item()] = file

    for i, file in enumerate(black_files):
        pic = torchvision.io.read_image(file).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        black_results[i] = score.item()
        if (score.item() in black_dict.keys()):
            print(black_dict[score.item()])
            print(file,"\n")
        black_dict[score.item()] = file
    

    for i, file in enumerate(white_files):
        pic = torchvision.io.read_image(file).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        white_results[i] = score.item()
        if (score.item() in white_dict.keys()):
            print(white_dict[score.item()])
            print(file,"\n")
        white_dict[score.item()] = file

    print("Asian Men Short Hair Mean:", asian_results.mean())
    print("Black Men Short Hair Mean:", black_results.mean())
    print("White Men Short Hair Mean:", white_results.mean())

    print("Asian Men Short Hair StdDev:", asian_results.std())
    print("Black Men Short Hair StdDev:", black_results.std())
    print("White Men Short Hair StdDev:", white_results.std())

    # plot all scores of images
    asian_results_sorted = np.sort(asian_results)
    black_results_sorted = np.sort(black_results)
    white_results_sorted = np.sort(white_results)

    x = np.linspace(1,100,100)

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, asian_results_sorted, label="Asian")
    plt.plot(x, black_results_sorted, label="Black")
    plt.plot(x, white_results_sorted, label="White")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/TypicalFacesScores.png")
    plt.clf()

    # plot lowest scoring images
    lowest_images = []
    for i in range(10):
        lowest_images.append(asian_dict[asian_results_sorted[i]])
    for i in range(10):
        lowest_images.append(black_dict[black_results_sorted[i]])
    for i in range(10):
        lowest_images.append(white_dict[white_results_sorted[i]])

    show_images(lowest_images, "images/AtypicalFaces/plots/LowestScoresTypical", "10 Lowest Scoring Typical Faces by Race")
    plt.cla()

    # plot highest scoring images
    highest_images = []
    for i in range(99,89,-1):
        highest_images.append(asian_dict[asian_results_sorted[i]])
    for i in range(99,89,-1):
        highest_images.append(black_dict[black_results_sorted[i]])
    for i in range(99,89,-1):
        highest_images.append(white_dict[white_results_sorted[i]])

    show_images(highest_images, "images/AtypicalFaces/plots/HighestScoresTypical", "10 Highest Scoring Typical Faces by Race")
    plt.cla()

    # create dataframe with image, score, race, hair, average rgb
    asian_df = pd.DataFrame.from_dict(asian_dict, orient='index').reset_index()
    asian_df = asian_df.rename(columns={0: 'image', 'index':'discriminator_score'})
    asian_df['race'] = 'asian'
    asian_df['hair_length'] = 'short'

    black_df = pd.DataFrame.from_dict(black_dict, orient='index').reset_index()
    black_df = black_df.rename(columns={0: 'image', 'index':'discriminator_score'})
    black_df['race'] = 'black'
    black_df['hair_length'] = 'short'

    white_df = pd.DataFrame.from_dict(white_dict, orient='index').reset_index()
    white_df = white_df.rename(columns={0: 'image', 'index':'discriminator_score'})
    white_df['race'] = 'white'
    white_df['hair_length'] = 'short'

    df = pd.concat([asian_df, black_df, white_df])

    r = []
    g = []
    b = []

    for i, row in df.iterrows():
        red_mean, green_mean, blue_mean = get_average_color_image(row['image'])
        r.append(red_mean)
        g.append(green_mean)
        b.append(blue_mean)

    df["red_mean"] = r
    df["green_mean"] = g
    df["blue_mean"] = b

    lab = []
    for i, row in df.iterrows():
        path = row['image']
        path = path.replace("rbg_images", "lab_images")
        image = cv2.imread(path)
        luminance = image[:,:,0]
        mean = np.mean(np.array(luminance))
        lab.append(mean)
    df["luminance"] = lab
    
    return asian_results_sorted, black_results_sorted, white_results_sorted, df
    
    
def main_combined():
    asian_long, black_long, white_long, atypical_df = main_atypical()
    asian_short, black_short, white_short, typical_df = main_typical()

    complete_df = pd.concat([atypical_df, typical_df])
    print(complete_df.shape)
    complete_df.to_csv("images/AtypicalFaces/data/AtypicalFaceData.csv")

    x = np.linspace(1,100,100)

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, asian_long, label="Asian Long")
    plt.plot(x, black_long, label="Black Long")
    plt.plot(x, white_long, label="White Long")
    plt.plot(x, asian_short, label="Asian Short")
    plt.plot(x, black_short, label="Black Short")
    plt.plot(x, white_short, label="White Short")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/CombinedFacesScores.png")
    plt.clf()

    # create histograms
    fig = plt.figure(figsize=(6.4, 4.8))
    plt.hist(asian_long, bins=100, alpha=0.5, label="asian long")
    plt.hist(black_long, bins=100, alpha=0.5, label=" black long")
    plt.hist(white_long, bins=100, alpha=0.5, label=" black long")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Atypical Score Distributions")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/AtypicalDistributions.png")
    plt.clf()

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.hist(asian_short, bins=100, alpha=0.5, label="asian short")
    plt.hist(black_short, bins=100, alpha=0.5, label=" black short")
    plt.hist(white_short, bins=100, alpha=0.5, label=" white short")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Typical Score Distributions")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/TypicalDistributions.png")
    plt.clf()

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.hist(asian_long, bins=100, alpha=0.5, label="asian long")
    plt.hist(asian_short, bins=100, alpha=0.5, label=" asian short")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Asian Score Distributions")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/AsianDistributions.png")
    plt.clf()

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.hist(black_long, bins=100, alpha=0.5, label="black long")
    plt.hist(black_short, bins=100, alpha=0.5, label="black short")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Black Score Distributions")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/BlackDistributions.png")
    plt.clf()

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.hist(white_long, bins=100, alpha=0.5, label="white long")
    plt.hist(white_short, bins=100, alpha=0.5, label=" white short")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("White Score Distributions")
    plt.legend()
    plt.savefig("images/AtypicalFaces/plots/WhiteDistributions.png")
    plt.clf()


def main():
    #convert_rbg_to_lab("ffhq_images_1024x1024/*","ffhq_images_1024x1024_lab_format/*")
    #read_images_get_scores()
    dict_files = glob.glob("parsing_dictionaries/rgb/*") 
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
    print("Correlation for all image is:", get_correlation(files, complete_dict))

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



def plot_normal_quantile(input):
    sm.qqplot(input, line="45")
    plt.savefig("images/normal_qq_plot.png")

def show_images(files, figurename, title):
    images = []
    for file in files:
        # name = glob.glob("ffhq_images_1024x1024_lab_format/*/"+str(file))
        image = Image.open(file)
        images.append(image)

        
    fig = plt.figure(figsize=(20, 8))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 10),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 share_all=True,
                 )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
        ax.imshow(im)
    
    # for i in range(len(grid.axes_all)):
    #     grid.axes_all[i].set_title(i)

    grid.axes_all[0].set_ylabel("Asian", fontsize=18)
    grid.axes_all[10].set_ylabel("Black", fontsize=18)
    grid.axes_all[20].set_ylabel("White", fontsize=18)

    fig.suptitle(title, fontsize=30)

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
    print("converting")
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
            file_name = write_folder+"/"+splits[-1]
            LAB_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            cv2.imwrite(file_name,LAB_image)

# bad generated images in out directory: 5195, 5196, 5197, 5198, 5200, 5201, 

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

def get_average_color_image(file):
    image = torchvision.io.read_image(file)
    image  = image.float()
    red_mean, green_mean, blue_mean = torch.mean(image,dim=[1,2]).numpy()
    return red_mean, green_mean, blue_mean


if __name__=="__main__":
    # convert_rbg_to_lab("images/AtypicalFaces/rbg_images/*", "images/AtypicalFaces/lab_images/*")
    main_combined()
    
