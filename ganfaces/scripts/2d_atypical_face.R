lib<-c("lme4","readr","tidyr","effects","ggplot2","psych","MASS",
       "Rmisc","plyr","dplyr","lmerTest","ggthemes","lsmeans",
       "pastecs","sjstats","car","readxl","ggdist", "svglite","gghdr", "ggdensity")
lapply(lib,require,character.only=TRUE)

gan_hair<-read.csv("data/AtypicalFaceData.csv")

mean_gan_hair<- summarySE(data=gan_hair, measurevar = "discriminator_score",
                          groupvars = c("race","hair_length"),
                          na.rm = FALSE, conf.interval = 0.95)
                          


plot <- ggplot(gan_hair, aes(discriminator_score, luminance)) +
    geom_point(shape = 21) +
     geom_hdr()

## plot <- ggplot(gan_hair, aes(discriminator_score, luminance)) +
##     geom_point(shape = 21) +
##     geom_hdr()

race_plot <- ggplot(gan_hair, aes(discriminator_score, luminance, fill=race)) +
    geom_point(shape = 21) +
    geom_hdr()


ggsave("results/figures/2d_plot.jpg", plot=race_plot)

