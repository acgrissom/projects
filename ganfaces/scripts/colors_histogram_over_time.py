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
sns.set() # Setting seaborn as default style even if use only matplotlib
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
         histogram = seaborn_plot_histogram_bin_by_score(df,
                                                        filename_prefix,
                                                        num_bins=num_bins,
                                                        out_dir=out_img_dir,
                                                        save_file=True)



def plot_over_time_facet(in_csv_dir : str,
                         out_img_dir : str,
                         start_iteration=0,
                         end_iteration=25000,
                         num_bins=20,
                         step=5000,
                         model_restrictions : list = None,
                         plot_type: str = "scatter",
                         figsize=(20,5)) -> None:
     num_figures = (end_iteration - start_iteration) / step + 1
     num_models = len(model_restrictions)
     figs, axes = plt.subplots(num_models,
                               int(num_figures),
                               sharex=False,
                               figsize=figsize
                               )
     print("Figures per row:",num_figures)
     print("Num. models:", num_models)
     

     # 1 row, num_figures columns, don't share axes
     figs.suptitle("Evolution of Color Bias During Training with DIfferent Seeds")
     if plot_type == "scatter":
          figs.supxlabel("Luminance")
          figs.supylabel("Score")
     else:
          figs.supxlabel("Score")
          figs.supylabel("Frequency (log)")
          

     files = glob.glob(in_csv_dir +"/*.csv", recursive=False)
     files = [f.replace("//","/") for f in files]             
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) <= end_iteration]
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) >= start_iteration]
     files = [f for f in files if int(Path(f).stem.split('-')[-1].replace(".pkl","")) % step == 0]
     # if model_restrictions is not None:
     #      for model_string in model_restrictions: 
     #           files = [f for f in files if model_string in f]
     # files = sorted(files)

     if model_restrictions is None:
          print("Files to be processed:")
          for f in files:
               print(f)
     model_files = []
     model_index = 0
     if num_models > 1:
          for model_string in model_restrictions:
               model_files.append([f for f in files if model_string in f])
               model_files[model_index] = sorted(model_files[model_index])
               print("Processing model", model_string)
               axis_counter = 0

               for csv_file in model_files[model_index]:
                    if axis_counter > num_figures - 1:
                         break
                    filename_prefix = Path(csv_file).stem
                    df = load_data(csv_file.strip())
                    append_hex_colors(df)
                    axis = axes[model_index, axis_counter]
                    print("Adding " +
                          str(model_index) + "," + str(axis_counter))
                    axis_counter += 1
                    if plot_type.lower() == "scatter":
                         #axis.set_xlabel("Luminance", fontsize=6)
                         #axis.set_ylabel("Score", fontsize=6)                         
                         scatter = plot_scatter(df,
                                                filename_prefix,
                                                out_dir=out_img_dir,
                                                save_file=False,
                                                axis=axis)
                    else:
                         #axis.set_xlabel("Score", fontsize=6)
                         #axis.set_ylabel("Freq. (log)", fontsize=6)
                         histogram = seaborn_plot_histogram_bin_by_score(df,
                                                                         filename_prefix,
                                                                         num_bins=num_bins,
                                                                         out_dir=out_img_dir,
                                                                         save_file=False,
                                                                         axis=axis)
               model_index += 1
          row_headers: list[str] =  ["Seed " + str(i) for i in range(len(model_files))]
          col_headers: list[str] = [str(x) if x == 0 else str(int(x / 1000)) + "K" for x in range(end_iteration + 1) if x % step == 0]

          for r, ax in zip(row_headers, axes[:, 0]):
               ax2 = ax.twinx()
               # move extra axis to the left, with offset
               ax2.yaxis.set_label_position('left')
               ax2.spines['left'].set_position(('axes', -0.4))
               # hide spine and ticks, set group label
               ax2.spines['left'].set_visible(False)
               ax2.set_yticks([])
               ax2.set_ylabel(r, rotation=0, size='large',
                              ha='right', va='center')
          
          for c, ax in zip(col_headers, axes[0]):
               ax.set_title(c, size='large')
          file_fullpath = out_img_dir + "/faceted_" + plot_type + "_test.png"
          print("Outputting to  ", out_img_dir)
          figs.savefig(file_fullpath, bbox_inches="tight")
          file_fullpath = file_fullpath.replace(".png",".svg")
          print("Outputting to  ", file_fullpath)
          figs.savefig(file_fullpath)

     else: # simple case, one model        
          axis_counter = 0
          print("Outputting to  ", out_img_dir)
          for csv_file in files:
              filename_prefix = Path(csv_file).stem
              df = load_data(csv_file.strip())
              append_hex_colors(df)
              #scatter = plot_scatter(df, filename_prefix, out_dir=out_img_dir, save_file=True)
              #https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8

              axis = axes[axis_counter]
              histogram = seaborn_plot_histogram_bin_by_score(df,
                                                              filename_prefix,
                                                              num_bins=num_bins,
                                                              out_dir=out_img_dir,
                                                              save_file=False,
                                                              axis=axis)
              #histogram.axes(ax=axes[axis_counter])
              axis_counter += 1
          file_fullpath = out_img_dir + "/" + plot_type + "_fig.png"
          print("Outputting to  ", file_fullpath)
          fig.savefig(file_fullpath)
          file_fullpath = file_fullpath.replace(".png",".svg")
          print("Outputting to  ", file_fullpath)
          fig.savefig(file_fullpath)

         



def main():
    parser = argparse.ArgumentParser( description='Creaes plots with average colors over training iterations')
    parser.add_argument('-b', '--num_bins', default=20, type=int,
    help="Number of bins for histogram. Default is 20.")
    parser.add_argument('-i', '--input_dir', type=str, required=True,
    help="Input directory of CSV files in correct format.")
    parser.add_argument('-o', '--output_dir', type=str, required=True,
    help="Output directory of generated plots.")
    parser.add_argument('-f', '--first_iteration', type=int, default=0, help="First iteration of model tranining to use. Default is 0.")
    parser.add_argument('-l', '--last_iteration', type=int,
    default=25000, help="Last iteration of model training t ouse. Default is 25000.")
    parser.add_argument('-s', '--step', default=1000, type=int, help="Step size between models to to  generate figures for.  Default is 1000.")
    parser.add_argument('-c', '--combine', default="False", type=str,  help="If 'True', genereates faceted plots.")
    parser.add_argument('-r', '--restrict', nargs='+', default=None, type=str)
    parser.add_argument('-p', '--plot_type',
    default="histogram", type=str, help="histogram or scatter")
                        
    args = parser.parse_args()
    NUM_BINS = args.num_bins
    COMBINE = args.combine
    OUT_DIR = os.path.dirname(args.output_dir)
    if COMBINE.lower() == "true":
         plot_over_time_facet(args.input_dir,
                              args.output_dir,
                              start_iteration=args.first_iteration,
                              end_iteration=args.last_iteration,
                              step=args.step,
                              num_bins=args.num_bins,
                              model_restrictions=args.restrict,
                              plot_type=args.plot_type
                              )
    else:
         plot_over_time(args.input_dir,
                        args.output_dir,
                        start_iteration=args.first_iteration,
                        end_iteration=args.last_iteration,
                        step=args.step,
                        num_bins=args.num_bins)
         
         
if __name__ == "__main__":
     main()
                        
                        
                        
