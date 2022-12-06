setwd("/Users/BaileyLin/Desktop")
data=read.csv("correct_LAB_format_images_data.csv",header=TRUE)
attach(data)
x1 = luminance
x2 = red_mean
x3 = green_mean
x4 = blue_mean
y = scores_rgb
model = lm(y~x1+x2+x3+x4)
summary(model)
library(car)
vif(model)
par(mfrow=c(2,2))
plot(model)
pairs(y~x1+x2+x3+x4)
mmps(model)
