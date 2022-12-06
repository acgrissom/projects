import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import glob
import torch
import click
import os
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    do_multiple_regression()
    

@click.command()
@click.option('--x', help='names of the dependent variables',default = "luminance",show_default=True)
@click.option('--y', help='name of the response variable',default = "scores_rgb",show_default=True)
def plot_regression(x,y):
    x_names = list(x)
    y_names = y
    data = pd.read_csv("correct_LAB_format_images_data.csv")
    sns.lmplot(data=data, x=x_names,y=y_names, fit_reg=True, 
               line_kws={"color":"red"}, scatter_kws={'alpha':0.3, "color":"green"})
    plt.savefig("regression.svg", format="svg")

def do_multiple_regression():
    df = pd.read_csv("correct_LAB_format_images_data.csv")
    X = df[["scores_rgb","red_mean","green_mean","blue_mean"]].to_numpy()
    y = df["luminance"].to_numpy()
    reg = LinearRegression().fit(X, y)
    print(f"The weights are:{reg.coef_}, and the intercept is:{reg.intercept_}")
    print(f"R^2 is {reg.score(X, y)}")

def plot_colored_hist():
    pass

if __name__ == "__main__":
    main()