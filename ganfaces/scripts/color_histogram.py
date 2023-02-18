import matplotlib.pyplot as plt
import pyplot_themes as themes
import pandas as pd
from colormap import rgb2hex
from sklearn.cluster import KMeans
import seaborn as sns

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

def plot_scatter(df) -> tuple:
    plot = plt.figure()
    plt.style.use('ggplot')
    plt.scatter(df.luminance, df.scores_rgb, c=df.color_mean, alpha=0.6)
    plt.xlabel("Luminance", fontsize=10)  
    plt.ylabel("Score", fontsize=10)

    plt.savefig(OUT_DIR + 'training_color_scatter.svg')
    plt.savefig(OUT_DIR + 'training_color_scatter.jpg')
    plt.savefig(OUT_DIR + 'training_color_scatter.png', dpi=100)
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
    for cluster_num in range(6):
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
        cluster_colors[cluster_num] = hex_color
    return cluster_colors 


"""
Returns a list of hex values with avearge color, one for each bin.
"""
def find_average_color_by_bin(df, num_bins) -> list:
    bin_colors = list()
    labels = range(num_bins)
    df['score_bin'] = pd.cut(df['scores_rgb'], bins=num_bins, labels=labels)

    for bin_num in labels:
        current_bin = df.loc[df['score_bin'] == bin_num]
        bin_red = round(current_bin['red_mean'].mean())
        bin_green = round(current_bin['green_mean'].mean())
        bin_blue = round(current_bin['blue_mean'].mean())
        hex_color = rgb2hex(round(bin_red), round(bin_green), round(bin_blue))
        bin_colors.append(hex_color)
    return bin_colors
    

def plot_histogram_bin_by_color(df, num_bins=6):
    cluster_colors = find_cluster_colors(df)
    
    
def plot_histogram_bin_by_score(df, num_bins=6):
    bin_colors  = find_average_color_by_bin(df, num_bins)
    plt.figure(figsize=(3.5,3.5))
    plt.tick_params(labelleft=False, left=False)
    plt.xticks(fontsize=10)
    plot = plt.hist(df['scores_rgb'], bins=num_bins, label=None, log=True)
    (n, bins, patches) = plot
    print(bin_colors)
    for i in range(len(bin_colors)):
        patches[i].set_color(bin_colors[i])

    plt.xlabel("Frequency (log)", fontsize=10)  
    plt.ylabel("Count", fontsize=10)
    #plt.bar(range(num_bins), [5]*num_bins, color=bin_colors)
    #plt.legend(loc="lower left")
    #plt.legend(labelcolor='black')

                 
    plt.savefig(OUT_DIR + 'training_color_histogram_logscale.svg')
    plt.savefig(OUT_DIR + 'training_color_histogram_logscale.jpg')
    

def seaborn_plot_histogram_bin_by_score(df, num_bins=6):
    plt.style.use('ggplot')

    #plt.figure(figsize=(3.5,3.5))
    plt.figure()
    bin_colors  = find_average_color_by_bin(df, num_bins)
    plot = sns.histplot(df['scores_rgb'], bins=num_bins, log_scale=(False,True))
     # print(bin_colors)
    for i in range(len(bin_colors)):
        plot.patches[i].set_color(bin_colors[i])
    
    plt.xticks(fontsize=10)
    plt.tick_params(labelleft=False, left=False)
    plt.xlabel("Score", fontsize=10) 
    plt.ylabel("Frequency (log)", fontsize=10)


    #plt.bar(range(num_bins), [5]*num_bins, color=bin_colors)
    #plt.legend(loc="lower left")
    #plt.legend(labelcolor='black')

                 
    plt.savefig(OUT_DIR + 'training_color_histogram_logscale.svg')
    plt.savefig(OUT_DIR + 'training_color_histogram_logscale.jpg')

    
if __name__ == "__main__":
   #themes.theme_minimal()
    NUM_BINS = 20
    df = load_data()
    append_hex_colors(df)
    #plot_histogram_bin_by_color(df, num_bins=NUM_BINS):  unfinished
    plot_scatter(df)
    seaborn_plot_histogram_bin_by_score(df, num_bins=NUM_BINS)

    #plt.show()
    


