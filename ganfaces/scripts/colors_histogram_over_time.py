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
from color_histogram import *
from datetime import datetime

def plot_over_time(in_csv_dir : str,
                   out_img_dir : str,
                   start_iteration=0,
                   end_iteration=25000,
                   num_bins=20,
                   step=20):
     logfile = "logs/colors_histogram_over_time_" + datetime.now().isoformat() + ".log"
     logging.basicConfig(filename=logfile, level=logging.INFO)
     files = glob.glob(in_csv_dir +"/*.csv", recursive=False)
     files = [f.replace("//","/") for f in files]             
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) <= end_iteration]
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) >= start_iteration]
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) % step == 0]
     for csv_file in files:
         logging.info("Processing " + csv_file)
         filename_prefix = Path(csv_file).stem
         print("****out_dir", out_img_dir)
         df = load_data(csv_file.strip())
         logging.info(df.dtypes) 
         append_hex_colors(df)
         scatter = plot_scatter(df, filename_prefix, out_dir=out_img_dir, save_file=True)
         histogram =seaborn_plot_histogram_bin_by_score(df,
                                                        filename_prefix,
                                                        num_bins=num_bins,
                                                        out_dir=out_img_dir,
                                                        save_file=True)



def plot_over_time_facet(in_csv_dir : str,
                         out_img_dir : str,
                         start_iteration=0,
                         end_iteration=25000,
                         num_bins=20,
                         step=5000):
     files = glob.glob(in_csv_dir +"/*.csv", recursive=False)
     files = [f.replace("//","/") for f in files]             
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) <= end_iteration]
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) >= start_iteration]
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) % step == 0]
     g = sns.FacetGrid(df, col='country', hue='country', col_wrap=4, )
     for csv_file in files:
         filename_prefix = Path(csv_file).stem
         print("****out_dir", out_img_dir)
         df = load_data(csv_file.strip())
         append_hex_colors(df)
         scatter = plot_scatter(df, filename_prefix, out_dir=out_img_dir, save_file=True)
         histogram =seaborn_plot_histogram_bin_by_score(df,
                                                        filename_prefix,
                                                        num_bins=num_bins,
                                                        out_dir=out_img_dir,
                                                        save_file=True)



def main():
    parser = argparse.ArgumentParser(
        description='Plots binned histograms (by score) with average colors over training iterations')
    parser.add_argument('-b', '--num_bins', default=20, type=int,
                        help="Number of bins for histogram. Default is 20.")     
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Input directory of CSV files in correct format.")
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help="Output directory of generated histograms.")
    parser.add_argument('-f', '--first_iteration', type=str, default=0,
                        help="First iteration of model tranining to use. Default is 0.")
    parser.add_argument('-l', '--last_iteration', type=str,  default=25000,
                        help="Last iteration of model training t ouse.  Default is 25000.")
    parser.add_argument('-s', '--step', default=1000, type=int,
                        help="Step size between models to to generate figures for.  Default is 1000.")
    args = parser.parse_args()
    NUM_BINS = args.num_bins
    OUT_DIR = os.path.dirname(args.output_dir)
    plot_over_time(args.input_dir,
                   args.output_dir,
                   start_iteration=args.first_iteration,
                   end_iteration=args.last_iteration,
                   step=args.step,
                   num_bins=args.num_bins)

     
if __name__ == "__main__":
    main()
