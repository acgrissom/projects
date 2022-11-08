import torchvision
import numpy as np
import pickle
import cv2

with open('models/stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
    D = pickle.load(f)['D'].cuda()  # torch.nn.Module
resize = torchvision.transforms.Resize((1024,1024))

example=cv2.imread("images/seed10000.png")
convert=cv2.cvtColor(example,cv2.COLOR_BGR2LAB)
print(D(convert))
cv2.imwrite("images/test.png",convert)
test=cv2.imread()