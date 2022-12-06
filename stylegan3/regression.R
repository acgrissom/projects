data=read.csv("correct_LAB_format_images_data.csv",header=TRUE)
attach(data)
x1 = scores_rgb
x2 = red_mean
x3 = green_mean
x4 = blue_mean
y = luminance
model = lm(y~x1+x2+x3+x4)
summary(model)
install.packages("car",repo = "https://lib.ugent.be/CRAN")
library(car)
vif(model)

