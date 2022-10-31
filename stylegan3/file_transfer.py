import json
import shutil
import sys
import glob

def main():
    least_100, top_100 = get_top_least_100_file_names(rgb=False)
    image_format = "lab"
    top_100_dir = "top_least_100_images/top_100/"+image_format
    least_100_dir = "top_least_100_images/least_100/"+image_format
    dir = ""
    if image_format=="rgb":
        dir = "ffhq_images_1024x1024/*/"
    else:
        dir = "ffhq_images_1024x1024_lab_format/*/"
    
    try:
        for file in least_100:
            name = glob.glob(dir+str(file))
            name = name[0]
            shutil.copy(name, least_100_dir)
        
        for file in top_100:
            name = glob.glob(dir+str(file))
            name = name[0]
            shutil.copy(name, top_100_dir)
            
    except IOError as e:
        print("Unable to copy file. %s" % e)
        sys.exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        sys.exit(1)
    

def get_top_least_100_file_names(rgb=True):
    files = []
    if rgb:
        files = ["parsing_dictionaries/rgb/dictionary9.txt","parsing_dictionaries/rgb/dictionary19.txt",
                "parsing_dictionaries/rgb/dictionary29.txt","parsing_dictionaries/rgb/dictionary39.txt",
                "parsing_dictionaries/rgb/dictionary49.txt","parsing_dictionaries/rgb/dictionary59.txt",
                "parsing_dictionaries/rgb/dictionary69.txt"]
    else:
        files = ["parsing_dictionaries/lab/dictionary9.txt","parsing_dictionaries/lab/dictionary19.txt",
                "parsing_dictionaries/lab/dictionary29.txt","parsing_dictionaries/lab/dictionary39.txt",
                "parsing_dictionaries/lab/dictionary49.txt","parsing_dictionaries/lab/dictionary59.txt",
                "parsing_dictionaries/lab/dictionary69.txt"]
    dict_list = []
    for file in files:
        with open(file, 'r') as convert_file:
            dictionary = convert_file.read()
            dict_list.append(json.loads(dictionary))
    
    complete_dict = {}
    for one_dict in dict_list:
        complete_dict = complete_dict | one_dict
    complete_dict = dict(sorted(complete_dict.items(), key=lambda item: item[1]))
    
    files = list(complete_dict.keys())
    least_100 = files[0:100]
    top_100 = files[69900:]
    
    return least_100, top_100
    

if __name__ == "__main__":
    main()