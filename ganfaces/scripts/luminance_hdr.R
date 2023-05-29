library(ggplot2)
library(ggdensity)
library(tools)
# get command line argument for input filename
# First is input file; second is output prefix

args = commandArgs(trailingOnly=TRUE)
csv_full_path = ""
output_full_path = ""

if (length(args) < 2) {
  stop("You must specify the input (csv) and output filename prefix (svg).", call.=FALSE)
} else if (length(args) == 1) {
                                        # default output file
    csv_full_path = args[1]
    output_full_path = "results/figures/" + tools::file_path_sans_ext(output_filename(csv_full_path)) + ".svg"
} else if(length(args) == 2) {
    csv_full_path = args[1]
    output_full_path = args[2]
}

output_filename = basename(output_full_path)
output_full_path_no_extension = tools::file_path_sans_ext(output_full_path)



                                        #data=read.csv("data/correct_LAB_format_images_data.csv",header=TRUE)
data = read.csv(csv_full_path)
attach(data)
x1 = luminance
x2 = red_mean
x3 = green_mean
x4 = blue_mean
y = scores_rgb
#theme_set(theme_minimal())
p_points <- ggplot(data, aes(data, x=luminance, y=scores_rgb)) +
    xlab("Luminance") + ylab("Score") +   labs(fill='HDR') +
    geom_point() + theme(legend.position = "bottom", legend.title = element_text("HDR")) 

p_hdr_points <- ggplot(data, aes(x=luminance, y=scores_rgb, xlab="Luminance", ylab="Score")) +
    xlab("Luminance") + ylab("Score") +  labs(fill='HDR') +
    geom_hdr_points() + theme(legend.position = "bottom", legend.title = element_text("HDR")) +
    geom_smooth(method='lm', se=FALSE) 

#p_points + p_hdr_points

#ggsave("results/figures/luminance_hdr.svg")
#ggsave("results/figures/luminance_hdr.jpg")
output_hdr_svg = paste(output_full_path_no_extension, "_luminance_hdr.svg", sep="")
print(paste("Writing", output_hdr_svg))
ggsave(paste(output_full_path_no_extension, "_luminance_hdr.svg", sep=""))
ggsave(paste(output_full_path_no_extension, "_luminance_hdr.jpg", sep=""))

luminance_dists <- ggplot(data, aes(x=luminance)) +
    xlab("Luminance") +  
    geom_density() + theme(legend.position = "bottom") +
    geom_hdr_rug()

#ggsave("results/figures/luminance_dist.svg")
                                        #ggsave("results/figures/luminance_dist.jpg")

output_hdr_dist = paste(output_full_path_no_extension, "_luminance_hdr.svg", sep="")
print(paste("Writing", output_hdr_dist))

ggsave(paste(output_full_path_no_extension, "_luminance_dist.svg", sep=""))
ggsave(paste(output_full_path_no_extension, "_luminance_dist.jpg", sep=""))
