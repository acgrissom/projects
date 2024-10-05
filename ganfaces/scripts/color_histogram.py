import matplotlib.pyplot as plt
import pandas as pd
from colormap import rgb2hex
from sklearn.cluster import KMeans
import seaborn as sns
import os
from pathlib import Path
import argparse
import glob
import logging
Path('/root/dir/sub/file.ext').stem
OUT_DIR = 'results/figures/'
def load_data(filename='data/correct_LAB_format_images_data.csv'):
    return pd.read_csv(filename)

def append_hex_colors(df):
    hex_list = []   
    for index, row in df.iterrows():
        r = row['red_mean']
        g = row['green_mean']
        b = row['blue_mean']
        hex_val = rgb2hex(round(r), round(g), round(b))
        hex_list.append(hex_val)
    df['color_mean'] = hex_list


def plot_scatter(df, filename_prefix, out_dir=OUT_DIR, save_file=True, axis=None) -> tuple:
    plot = None
    if axis is not None:
        plot = axis.scatter(df.luminance, df.scores_rgb, c=df.color_mean, alpha=0.6, marker='s')
        plt.axis="off"
        plot.axis="off"       
        #plt.setp(axis, xlabel="Luminance", ylabel=None) #this removes axes successfully
        axis.set_xticks([])
        axis.set_yticks([])
        #plt.xlabel("Luminance", fontsize=12)  
        #plt.ylabel("Score", fontsize=12)

    else:
        plt.scatter(df.luminance, df.scores_rgb, c=df.color_mean, alpha=0.6, marker='s', axis=axis)
        plt.xlabel("Luminance", fontsize=12)  
        plt.ylabel("Score", fontsize=12)

        
    plot = plt.figure(figsize=(7,7))
    plt.style.use('ggplot')

    
   
    ##Delete following if code breaks
    fig, ax = plt.subplots(sharex=True, sharey=True)
    #plt.gca().set_xticklabels([])
    #plt.gca().set_yticklabels([])

    ax.set(xticklabels=[])  # remove the tick labels
    ax.set(yticklabels=[])  # remove the tick labels
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ax.set_axis_off()

    #####

    #plt.savefig(OUT_DIR + 'training_color_scatter.svg') #takes too long/too big
    if save_file == True:
        plt.savefig(out_dir + "/" +  filename_prefix + '_color_scatter.jpg')
        plt.savefig(out_dir + "/" + filename_prefix + '_color_scatter.png', dpi=100)
    return plot



"""
Returns dictionary of the average color of a cluster
"""
def find_cluster_colors(df, num_clusters=6) -> dict:
    kmeans = KMeans(num_clusters)
    X = df[['red_mean', 'blue_mean', 'green_mean']]
    y = kmeans.fit(X)
    df['color_cluster'] = y
    
    ### now find average color of cluster
    cluster_colors = {}
    for cluster_num in range(num_clusters):
        current_cluster = df.loc[df['color_cluster'] == cluster_num]
        cluster_red = round(current_cluster['red_mean'].mean())
        cluster_green= round(current_cluster['green_mean'].mean())
        cluster_blue = round(current_cluster['blue_mean'].mean())
        hex_color = rgb2hex(round(cluster_red), round(cluster_green), round(cluster_blue))
        cluster_colors[cluster_num] = hex_color
    return cluster_colors 

def find_score_average_color(df, num_clusters=6) -> dict:
    kmeans = KMeans(num_clusters)
    X = df[['red_mean', 'blue_mean', 'green_mean']]
    y = kmeans.fit(X)
    df['color_cluster'] = y
    
    ### now find average color of cluster
    cluster_colors = {}
    for cluster_num in range(num_clusters):
        current_cluster = df.loc[df['color_cluster'] == cluster_num]
        cluster_red = round(current_cluster['red_mean'].mean())
        cluster_green= round(current_cluster['green_mean'].mean())
        cluster_blue = round(current_cluster['blue_mean'].mean())
        hex_color = rgb2hex(round(cluster_red), round(cluster_green), round(cluster_blue))
        cluster_colors['cluster_red'] = cluster_red
        cluster_colors['cluster_green'] = cluster_green
        cluster_colors['cluster_blue'] = cluster_blue
        cluster_colors[cluster_num] = hex_color
    return cluster_colors 


"""
Returns a list of hex values with avearge color, one for each bin.
"""
def find_average_color_by_bin(df, num_bins) -> list:
    logging.info("find_average_color_by_bin()")
    logging.info("num_bins " + str(num_bins))
    bin_colors = list()
    labels = range(num_bins)
    df['score_bin'] = pd.cut(df['scores_rgb'], bins=num_bins, labels=labels)
    logging.info("df head after binning:\n" + df.to_string(max_rows=10))
    for bin_num in labels:
        current_bin = df.loc[df['score_bin'] == bin_num]
        if current_bin.empty:
            #bar of 0 height; so, color doesn't matter
            bin_red = bin_green = bin_blue = 0
        else:
            bin_red = round(current_bin['red_mean'].mean())
            bin_green = round(current_bin['green_mean'].mean())
            bin_blue = round(current_bin['blue_mean'].mean())

        # bin_colors['bin_red'] = bin_red
        # bin_colors['bin_green'] = bin_green
        # bin_colors['bin_blue'] = bin_blue

        hex_color = rgb2hex(round(bin_red), round(bin_green), round(bin_blue))
        bin_colors.append(hex_color)
    return bin_colors
    

def add_mean_color_per_image(df):
    mean_colors = list() 
    for idx, row in df.iterrows():
        r = float(row.loc['red_mean'])
        g = float(row.loc['green_mean'])
        b = float(row.loc['blue_mean'])
        mean_color =  rgb2hex(round(r), round(g), round(b))
        mean_colors.append(mean_color)
    df['mean_color'] = mean_colors
        

def plot_histogram_bin_by_color(df, num_bins=6):
    cluster_colors = find_cluster_colors(df)
    

def plot_histogram_bin_by_score(df,
                                num_bins=6,
                                out_dir=OUT_DIR,
                                save_file=True,
                                axes=None):
    bin_colors  = find_average_color_by_bin(df, num_bins)
    plt.figure(figsize=(7, 7))
    plt.tick_params(labelleft=False, left=False)
    plt.xticks(fontsize=12)
    plot = plt.hist(df['scores_rgb'], bins=num_bins, label=None, log=True)
    (n, bins, patches) = plot
    print(bin_colors)
    for i in range(len(bin_colors)):
        patches[i].set_color(bin_colors[i])

    plt.xlabel("Frequency (log)", fontsize=12)  
    plt.ylabel("Score", fontsize=12)
    #plt.bar(range(num_bins), [5]*num_bins, color=bin_colors)
    #plt.legend(loc="lower left")
    #plt.legend(labelcolor='black')
    if save_file == True:
        plt.savefig(out_dir + 'training_color_histogram_logscale.svg')
        plt.savefig(out_dir + 'training_color_histogram_logscale.jpg')
    return plot
    



def seaborn_plot_histogram_bin_by_score(df,
                                        filename_prefix,
                                        num_bins=6,
                                        out_dir=OUT_DIR,
                                        save_file=True,
                                        axis=None):
    
    logging.info("CSV scores_rgb_hasnull::"+str(df['scores_rgb'].isnull().values.any()))
    logging.info("CSV red_mean_hasnull::"+str(df['red_mean'].isnull().values.any()))
    logging.info("CSV green_mean_hasnull::"+str(df['green_mean'].isnull().values.any()))
    logging.info("CSV blue_mean_hasnull::"+str(df['blue_mean'].isnull().values.any()))
    plt.style.use('ggplot')
    plt.figure(figsize=(7 , 7))
    bin_colors  = find_average_color_by_bin(df, num_bins)
    plot = sns.histplot(df['scores_rgb'],
                        bins=num_bins,
                        log_scale=(False,True),
                        ax=axis)
    plt.setp(axis, xlabel="Score", ylabel="Frequency (log)") #this removes axes successfully
    plot.set(xticklabels=[])  # remove the tick labels
    plot.set(yticklabels=[])  # remove the tick labels

    if axis is not None: #faceted
        plt.setp(axis, xlabel=None, ylabel=None) #this removes axes successfully
        plot.set(xlabel=None)
        plot.set(ylabel=None)
    else:
        plt.xlabel("Score", fontsize=12) 
        plt.ylabel("Frequency (log)", fontsize=12)

     # print(bin_colors)
    for i in range(len(bin_colors)):
        plot.patches[i].set_color(bin_colors[i])
    
    #plt.xticks(fontsize=10)
    #plt.tick_params(labelleft=False, left=False)
  

    plot.tick_params(bottom=False)
    
    #plt.bar(range(num_bins), [5]*num_bins, color=bin_colors)
    #plt.legend(loc="lower left")
    #plt.legend(labelcolor='black')
    #fig, ax = plt.subplots()
    plt.tick_params(bottom=False)
    fig, ax = plt.subplots()
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    ax.set(xticklabels=[])  # remove the tick labels
    ax.set(yticklabels=[])  # remove the tick labels
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ax.set_axis_off()
    if(save_file == True):
        plt.savefig(out_dir + "/" + filename_prefix + "_color_histogram_logscale.svg")
        plt.savefig(out_dir + "/" + filename_prefix + "_color_histogram_logscale.jpg")
    return plot


"""Doesn't work."""
def make_density_estimation(df, num_bins=6):
    add_mean_color_per_image(df)
    plt.figure()
    #bin_colors  = find_average_color_by_bin(df, num_bins)
    f, ax = plt.subplots(figsize=(7, 7))
    x = x=df['red_mean']
    y = df['blue_mean']
    z = df['scores_rgb']
    sns.scatterplot(x=x, y=y, s=5, hue=z)
    #sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.histplot(x=x, y=y, bins=num_bins, pthresh=.1, hue=z)
    #sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
    #plt.savefig('density.svg')
    #plt.savefig('density.png')


"""
This doesn't look the way it should.
"""
def make_marginal_hexplot(df, num_bins=6):
    add_mean_color_per_image(df)
    plt.figure()
    bin_colors  = find_average_color_by_bin(df, num_bins)
    f, ax = plt.subplots(figsize=(7, 7))
    x = x=df['red_mean']
    y = df['blue_mean']
    z = df['scores_rgb']
    print(bin_colors)
    sns.color_palette("mako", as_cmap=True)
    sns.jointplot(x=x, y=y, hue=z, dropna=True)

   # plt.savefig('marginal.svg')
   # plt.savefig('marginal.png')


    
    
   
if __name__ == "__main__":
   #themes.theme_minimal()
    parser = argparse.ArgumentParser(
        description='Plots binned histograms (by score) with average colors')
    parser.add_argument('-b', '--num_bins', default=20, type=int)     
    parser.add_argument('-i', '--input_csv', type=str)
    parser.add_argument('-o', '--output_filename', type=str)
    args = parser.parse_args()
    NUM_BINS = args.num_bins
    OUT_DIR = os.path.dirname(args.output_filename)
    filename_prefix = Path(args.output_filename).stem
    print("Opening", args.input_csv)
    print("Outputting to", OUT_DIR)
    df = load_data(args.input_csv.strip())
    append_hex_colors(df)
    plot_scatter(df, filename_prefix)
    seaborn_plot_histogram_bin_by_score(df, filename_prefix, num_bins=NUM_BINS)

    #plt.show()
    


