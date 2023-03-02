#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob

df = pd.read_csv("correct_LAB_format_images_data.csv", index_col=False)

def vis_parsing_maps(im, parsing_anno, file,stride,):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    unique_parsing = np.unique(parsing_anno)
    # vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    # num_of_class = np.max(vis_parsing_anno)

    # for pi in range(1, num_of_class + 1):
    #     index = np.where(vis_parsing_anno == pi)
    #     vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    # vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    
 
    # locations = np.logical_or(vis_parsing_anno==1, vis_parsing_anno==10)
    locations1 = (vis_parsing_anno==0)
    locations2 = (vis_parsing_anno==1)
    locations3 = (vis_parsing_anno==17)
    locations4 = np.logical_or(vis_parsing_anno==1, vis_parsing_anno==10)
    locations5 = (vis_parsing_anno==1)+(vis_parsing_anno==10)+(vis_parsing_anno==4)+(vis_parsing_anno==5)
    temp = im[:,:,0]*0.212671+im[:,:,1]*0.715160+im[:,:,2]*0.072169
    mean_r1 = -1
    mean_g1 = -1
    mean_b1 = -1
    mean_luminance1=-1
    mean_r2 = -1
    mean_g2 = -1
    mean_b2 = -1
    mean_luminance2=-1
    mean_r3 = -1
    mean_g3 = -1
    mean_b3 = -1
    mean_luminance3=-1
    mean_r4 = -1
    mean_g4 = -1
    mean_b4 = -1
    mean_luminance4=-1
    mean_r5 = -1
    mean_g5 = -1
    mean_b5 = -1
    mean_luminance5=-1
    
    if(0 in unique_parsing):
        mean_r1 = np.mean(im[:, :, 0][locations1])
        mean_g1 = np.mean(im[:, :, 1][locations1])
        mean_b1 = np.mean(im[:, :, 2][locations1])
        mean_luminance1 = np.mean(temp[locations1])
    
    if(1 in unique_parsing):
        mean_r2 = np.mean(im[:, :, 0][locations2])
        mean_g2 = np.mean(im[:, :, 1][locations2])
        mean_b2 = np.mean(im[:, :, 2][locations2])
        mean_luminance2 = np.mean(temp[locations2])
    
    if(17 in unique_parsing):
        mean_r3 = np.mean(im[:, :, 0][locations3])
        mean_g3 = np.mean(im[:, :, 1][locations3])
        mean_b3 = np.mean(im[:, :, 2][locations3])
        mean_luminance3 = np.mean(temp[locations3])
        
    if(1 in unique_parsing and 10 in unique_parsing):
        mean_r4 = np.mean(im[:, :, 0][locations4])
        mean_g4 = np.mean(im[:, :, 1][locations4])
        mean_b4= np.mean(im[:, :, 2][locations4])
        mean_luminance4 = np.mean(temp[locations4])
        
    if(1 in unique_parsing and 10 in unique_parsing and 4 in unique_parsing and 5 in unique_parsing):
        mean_r5 = np.mean(im[:, :, 0][locations5])
        mean_g5 = np.mean(im[:, :, 1][locations5])
        mean_b5= np.mean(im[:, :, 2][locations5])
        mean_luminance5 = np.mean(temp[locations5])
    
    
    # vis_parsing_anno = (vis_parsing_anno==1).astype(int)+\
    #                     (vis_parsing_anno==10).astype(int)
                
                        
    # vis_parsing_anno = 255*(vis_parsing_anno - np.min(vis_parsing_anno))/(np.max(vis_parsing_anno)-np.min(vis_parsing_anno))
    #plt.imsave(save_path,test,cmap="gray")
    # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
    # cv2.imwrite(save_path, vis_parsing_anno, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return mean_r1,mean_g1,mean_b1, mean_luminance1, mean_r2,mean_g2, mean_b2, mean_luminance2,\
        mean_r3, mean_g3, mean_b3, mean_luminance3, mean_r4, mean_g4, mean_b4, mean_luminance4, \
        mean_r5,mean_g5, mean_b5, mean_luminance5 


def evaluate(cp='model_final_diss.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    rmean1=[]
    gmean1=[]
    bmean1=[]
    luminance1=[]
    
    rmean2=[]
    gmean2=[]
    bmean2=[]
    luminance2=[]
    
    rmean3=[]
    gmean3=[]
    bmean3=[]
    luminance3=[]
    
    rmean4=[]
    gmean4=[]
    bmean4=[]
    luminance4=[]
    
    rmean5=[]
    gmean5=[]
    bmean5=[]
    luminance5=[]
    
    
    
    image_ids = df["image_id"]
    folder = "ffhq_images_1024x1024/*/"
    
    with torch.no_grad():
        for index,id in enumerate(image_ids):
            file = glob.glob(folder+id)[0]
            img = Image.open(file)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(np.unique(parsing))
            r1,g1,b1,l1,r2,g2,b2,l2,r3,g3,b3,l3,r4,g4,b4,l4,r5,g5,b5,l5=vis_parsing_maps(image, parsing,file,stride=1)
            if(index%100==0):
                print("processed "+str(index)+" images")
            rmean1.append(r1)
            gmean1.append(g1)
            bmean1.append(b1)
            luminance1.append(l1)
            
            rmean2.append(r2)
            gmean2.append(g2)
            bmean2.append(b2)
            luminance2.append(l2)
            
            rmean3.append(r3)
            gmean3.append(g3)
            bmean3.append(b3)
            luminance3.append(l3)
            
            rmean4.append(r4)
            gmean4.append(g4)
            bmean4.append(b4)
            luminance4.append(l4)
            
            rmean5.append(r5)
            gmean5.append(g5)
            bmean5.append(b5)
            luminance5.append(l5)
    
    df["background_red_mean"]=rmean1
    df["background_green_mean"]=gmean1
    df["background_blue_mean"]=bmean1
    df["background_luminance"]=l1
    
    df["skin_red_mean"]=rmean2
    df["skin_green_mean"]=gmean2
    df["skin_blue_mean"]=bmean2
    df["skin_luminance"]=l2
    
    df["hair_red_mean"]=rmean3
    df["hair_green_mean"]=gmean3
    df["hair_blue_mean"]=bmean3
    df["hair_luminance"]=l3
    
    df["skin_nose_red_mean"]=rmean4
    df["skin_nose_green_mean"]=gmean4
    df["skin_nose_blue_mean"]=bmean4
    df["skin_nose_luminance"]=luminance4
    
    df["skin_nose_eyes_red_mean"]=rmean5
    df["skin_nose_eyes_rgreen_mean"]=gmean5
    df["skin_nose_eyes_rblue_mean"]=bmean5
    df["skin_nose_eyes_rluminance"]=luminance5
    
    df.to_csv("correct_LAB_format_images_data.csv",index=False)
    
    






if __name__ == "__main__":
    evaluate(cp='models/79999_iter.pth')


