import numpy as np
import pandas as pd
import cv2

import pickle
import torch
import torch.nn as nn
import torchvision

from PIL import Image
# import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
    D = pickle.load(f)['D'].cuda()  # torch.nn.Module
resize = torchvision.transforms.Resize((1024,1024))

def shuffle_images(csv):
    df = pd.read_csv(csv)
    shuffled_images = []
    scores = []

    i = 0
    for index, row in df.iterrows():
        filename = row["image"]
        image = cv2.imread(filename)
        oldImg = np.copy(image)

        rdmImg = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
        np.random.shuffle(rdmImg)
        shuffledImg = np.reshape(rdmImg, image.shape)
        
        splitname = filename.split("/")
        splitname[2] = "shuffled_images"
        newname = "/".join(splitname)

        cv2.imwrite("images/AtypicalFaces/shuffled_images/testImgOld.png", oldImg)
        cv2.imwrite(newname, shuffledImg)

        shuffled_images.append(newname)

        if i % 10 == 0:
            print(i)
        i += 1

    df['shuffled_images'] = shuffled_images
    print(df.head())
    df.to_csv("images/AtypicalFaces/data/ShuffledWomenFaceData.csv")


def get_shuffled_scores(csv):
    df = pd.read_csv(csv)
    scores = []

    for index, row in df.iterrows():
        filename = row["shuffled_images"]

        pic = torchvision.io.read_image(filename).cuda()
        pic = resize(pic)
        score = D(pic[None,:,:,:],c=None).cuda()
        scores.append(score.item())


    df['shuffled_discriminator_score'] = scores
    df.to_csv("images/AtypicalFaces/data/ShuffledWomenFaceData.csv")


def combine_dfs():
    df1 = pd.read_csv("images/AtypicalFaces/data/AtypicalShuffledFaceData.csv")
    df2 = pd.read_csv("images/AtypicalFaces/data/ShuffledWomenFaceData.csv")

    dfnew = pd.concat([df1, df2]).reset_index(drop=True)

    dfnew.to_csv("images/AtypicalFaces/data/ShuffledCombinedData.csv")

def get_top_bottom_100():
    df = pd.read_csv("images/AtypicalFaces/data/ShuffledCombinedData.csv")
    dfsorted = df.sort_values(by=["shuffled_discriminator_score"]).reset_index()

    head = dfsorted.head(100)
    tail = dfsorted.tail(100)

    bot100 = head["shuffled_images"]
    top100 = tail["shuffled_images"]

    return top100, bot100
    

def show_images(files, figurename):
    images = []
    for file in files:
        # name = glob.glob("ffhq_images_1024x1024_lab_format/*/"+str(file))
        image = Image.open(file)
        images.append(image)

        
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.savefig(figurename)


if __name__=="__main__":
    # shuffle_images("images/AtypicalFaces/data/WomenFaceData.csv")
    # get_shuffled_scores("images/AtypicalFaces/data/ShuffledWomenFaceData.csv")
    # combine_dfs()
    top, bot = get_top_bottom_100()
    show_images(top, "images/AtypicalFaces/plots/top100shuffled.png")
    show_images(bot, "images/AtypicalFaces/plots/bot100shuffled.png")
    print("ok")