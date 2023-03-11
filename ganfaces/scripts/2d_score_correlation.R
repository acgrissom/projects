lib<-c("lme4","readr","tidyr","effects","ggplot2","psych","MASS",
       "Rmisc","plyr","dplyr","lmerTest","ggthemes","lsmeans",
       "pastecs","sjstats","car","readxl","ggdist", "svglite","gghdr", "ggdensity")
lapply(lib,require,character.only=TRUE)

#gan_hair<-read.csv("data/study2_google_cleaned.csv")
df<-read.csv("data/cleaned/Study2_50_aggregated.csv")
#df<-read.csv("data/cleaned/Study2_unbalanced_aggregated.csv")
## mean_gan_hair<- summarySE(data=gan_hair, measurevar = "discriminator_score",
##                           groupvars = c("stim.race","hair_length"),
##                           na.rm = FALSE, conf.interval = 0.95)
                          

mean_gan<- summarySE(data=df, measurevar = "discriminator_score",
                          groupvars = c("stim.race","stim.gen"),
                          na.rm = FALSE, conf.interval = 0.95)


plot <- ggplot(df,  aes(x=SkintoneMean,  y=red_mean, fill=discriminator_score,  group=discriminator_score)) +
    geom_bin2d(bins=20) +
    scale_fill_continuous(type = "viridis") +
    facet_grid(stim.race ~ stim.gen) +
    labs(x="Skintone (darkness)", y="Red", color="Score", fill="Score")

    
         
    #geom_jitter(x = red_mean, y =AfrocentricMean, color = discriminator_score)
    #geom_hdr()

## plot <- ggplot(gan_hair, aes(discriminator_score, luminance)) +
##     geom_point(shape = 21) +
##     geom_hdr()

#race_plot <- ggplot(gan_hair, aes(discriminator_score, luminance, fill=race)) +
#    geom_point(shape = 21) +
#    geom_hdr()


ggsave("results/figures/red_skin_grid.svg", plot=plot)
ggsave("results/figures/red_skin_grid.jpg", plot=plot)
