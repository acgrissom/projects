library(ggplot2)
library(ggdensity)
data=read.csv("data/correct_LAB_format_images_data.csv",header=TRUE)
attach(data)
x1 = luminance
x2 = red_mean
x3 = green_mean
x4 = blue_mean
y = scores_rgb
theme_set(theme_minimal())
p_points <- ggplot(data, aes(data, x=luminance, y=scores_rgb)) +
    xlab("Luminance") + ylab("Score") +   labs(fill='HDR') +
    geom_point() + theme(legend.position = "bottom", legend.title = element_text("HDR")) 

p_hdr_points <- ggplot(data, aes(x=luminance, y=scores_rgb, xlab="Luminance", ylab="Score")) +
    xlab("Luminance") + ylab("Score") +  labs(fill='HDR') +
    geom_hdr_points() + theme(legend.position = "bottom", legend.title = element_text("HDR")) +
    geom_smooth(method='lm', se=FALSE) 

#p_points + p_hdr_points
ggsave("results/figures/luminance_hdr.svg")
ggsave("results/figures/luminance_hdr.jpg")

luminance_dists <- ggplot(data, aes(x=luminance)) +
    xlab("Luminance") +  
    geom_density() + theme(legend.position = "bottom") +
    geom_hdr_rug()

ggsave("results/figures/luminance_dist.svg")
ggsave("results/figures/luminance_dist.jpg")

