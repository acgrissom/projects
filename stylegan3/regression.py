import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import glob
import torch


def plot_regression():
    data = pd.read_csv("correct_LAB_format_images_data.csv")
    sns.lmplot(data=data, x="luminance",y="scores_rgb", fit_reg=True, 
               line_kws={"color":"red"}, scatter_kws={'alpha':0.3, "color":"green"})
    plt.savefig("regression.svg", format="svg")

def get_average_color():
    df = pd.read_csv("correct_LAB_format_images_data.csv")
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
