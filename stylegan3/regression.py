import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("correct_LAB_format_images_data.csv")
    sns.lmplot(data=data, x="luminance",y="scores_rgb")
    plt.savefig("regression.png")
if __name__=="__main__":
    main()