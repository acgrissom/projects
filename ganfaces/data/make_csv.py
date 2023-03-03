import pandas as pd
import os

def main():
    """
    Note that one must have the correct pathways set up for this to run
    or you can just change them to where your files are
    """
    #maps we can use to translate image names
    oldnew_map, newold_map = make_maps("csvs/name_mappings.csv")

    #all pathways
    srcs = ["images/AWB",
                "images/AWL",
                "images/BWB",
                "images/BWL",
                "images/WWB",
                "images/WWL",
                "images/AML",
                "images/AMS",
                "images/BML",
                "images/BMS",
                "images/WML",
                "images/WMS",
                "images/top",
                "images/bottom"
                ]

    all_images = []
    for src in srcs:
        all_images.extend(os.listdir(src))


    #reading in csvs for each file
    atypical_df = pd.read_csv("csvs/AtypicalFaceData.csv")
    women_df = pd.read_csv("csvs/WomenFaceData.csv")
    topbot_df = pd.read_csv("csvs/bottom_top_100_data.csv")

    ratings_df = pd.read_csv("csvs/GAN Face Survey Master Cleaned.csv")

    #get dataframes for each image
    image_responses = {}
    for image in all_images:
        frame = ratings_df.loc[ratings_df["Image"] == image]
        frame.reset_index(drop=True, inplace=True) #just for fun
        responses = frame["Response"].to_list()
        responses = clean_numbers(responses) #get rid of text where numbers are
        image_responses[image] = responses



    #topbottom data
    topbot_srcs = ["images/top", "images/bottom"]

    topbot_data = generate_data(topbot_srcs, topbot_df, newold_map, [0, 2, 3, 4, 5])

    #men data
    men_srcs = ["images/AML", "images/AMS", "images/BML", "images/BMS", "images/WML", "images/WMS"]

    men_data = generate_data(men_srcs, atypical_df, newold_map, [1, 5, 6, 7, 8])



    #women data
    women_srcs = ["images/AWL", "images/AWB", "images/BWL", "images/BWB", "images/WWL", "images/WWB"]

    women_data = generate_data(women_srcs, women_df, newold_map, [2, 5, 6, 7, 8])

    all_data = []

    #making big matrix
    topbot_mat = compile_data(topbot_srcs, topbot_data, image_responses)
    men_mat = compile_data(men_srcs, men_data, image_responses)
    women_mat = compile_data(women_srcs, women_data, image_responses)

    all_data.extend(topbot_mat)
    all_data.extend(men_mat)
    all_data.extend(women_mat)


    cols = ["image", "discriminator_Score", "red_mean", "green_mean", "blue_mean", "luminance"]
    #max 16(!) raters * 9 ratings
    for i in range(1, 17):
        cols.append(f"Skintone {i}")
        cols.append(f"Race {i}")
        cols.append(f"RaceTextbox {i}")
        cols.append(f"Afrocentric {i}")
        cols.append(f"Eurocentric {i}")
        cols.append(f"Asiocentric {i}")
        cols.append(f"Masculinity {i}")
        cols.append(f"Femininity {i}")
        cols.append(f"Hair {i}")

    all_df = pd.DataFrame(all_data, columns=cols)

    all_df.to_csv("csvs/all_face_data.csv", index=False)




def compile_data(srcs, data, responses):
    """
    Take in pathways srcs, extracted data, and survey responses
    Combine them all into one row for each image
    Returns big matrix with all data on images in srcs folders
    """
    all_data = []
    for src in srcs:
        for image in os.listdir(src):
            row = [image]
            row.extend(data[image])
            row.extend(responses[image])
            all_data.append(row)

    return all_data


def generate_data(srcs, df, name_map, desired_indices):
    """
    Take images from folders srcs and use dataframe df to get data from files
    We use name_map to translate names, and desired_indices to filter out data
    desired indices are different for each file because they are formatted differently
    Make a dictionary that maps image to its extracted data
    """
    paths = os.listdir(srcs[0])
    for i in range(1, len(srcs)):
        paths.extend(os.listdir(srcs[i]))

    data_dict = {}
    for image in paths:
        data = read_data(image, df, name_map, desired_indices)
        data_dict[image] = data

    return data_dict

def read_data(image, df, name_map, desired_indices):
    """
    For image, find instance of that image in dataframe df
    Use name_map to translate and desired indices to clean
    returns the images extracted data
    """
    name = name_map[image]
    temp = df.loc[df["image"].str.contains(name)]
    data = temp.iloc[0].to_list()
    final_data = [] #we have to do it like this because the files aren't formatted the same way
    for i in  range(len(data)):
        if i in desired_indices:
            final_data.append(data[i])

    return final_data


def clean_numbers(responses):
    """
    Some numbers in responses are accompanied by something like (not at all)
    This function returns the list with these responses cleaned, so its only numbers or only words
    """
    for i in range(len(responses)):
        num = int(i/9) * 9
        #these are the indices of responses that have numbers
        if i in [0 + num, 3 + num, 4 + num, 5 + num, 6 + num, 7 + num, 8 + num]:
            if ' ' in str(responses[i]):
                responses[i] = responses[i].split(' ')[0]

    return responses

def make_maps(path):
    """
    Read in the maps csv and make maps to translate file names
    """
    df = pd.read_csv(path)
    olds = df["Old"].to_list()
    news = df["New"].to_list()

    names_mapf = {}
    names_mapb = {}
    for i in range(len(olds)):
        names_mapf[olds[i]] = news[i]

    for i in range(len(news)):
        names_mapb[news[i]] = olds[i]

    return names_mapf, names_mapb

main()

