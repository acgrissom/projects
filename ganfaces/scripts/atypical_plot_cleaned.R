lib<-c("lme4","readr","tidyr","effects","ggplot2","psych","MASS",
       "Rmisc","plyr","dplyr","lmerTest","ggthemes","lsmeans",
       "pastecs","sjstats","car","readxl","ggdist", "svglite","gghdr")
lapply(lib,require,character.only=TRUE)

gan_hair<-read.csv("data/cleaned/Study2_GoogleFaces_cleaned_50pergroup.csv")
gan_hair_unbalanced<-read.csv("data/cleaned/Study2_GoogleFaces_full_unbalanced.csv")
mean_gan_hair<- summarySE(data=gan_hair, measurevar = "discriminator_score",
                          groupvars = c("stim.race","hairlength", "stim.gen"),
                          na.rm = FALSE, conf.interval = 0.95)




race_hair<- ggplot(gan_hair, aes(x=discriminator_score, fill = hairlength)) +
    geom_density(alpha = 0.25)+
    facet_grid(stim.race ~ stim.gen) +
    #theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          #legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12, face="bold"),
          #legend.title=element_blank(),
          legend.text=element_text(size=12)) +
        scale_color_colorblind() +
  labs(y = "Density", x = "Score",
       fill = "Hair") 





ggsave("results/figures/atypical_all_50face_dist.svg", plot=race_hair, height = 5, width=14)
ggsave("results/figures/atypical_all_50face_dist.jpg", plot=race_hair, height=5, width=14)


race_hair_unbalanced<- ggplot(gan_hair_unbalanced, aes(x=discriminator_score, fill = hairlength)) +
    geom_density(alpha = 0.25)+
    facet_grid(stim.race ~ stim.gen) +
    #theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12, face="bold"), 
          legend.text=element_text(size=12)) +
          scale_color_colorblind() +
          labs(y = "Density", x = "Score",
          fill = "Hair") 


 

ggsave("results/figures/atypical_all_50face_unbalanced_dist.svg", plot=race_hair_unbalanced, width=7)
ggsave("results/figures/atypical_all_50face_unbalanced_dist.jpg", plot=race_hair_unbalanced, width=7)


race_hair_unbalanced<- ggplot(gan_hair_unbalanced, aes(x=discriminator_score, fill = hairlength)) +
    geom_histogram(alpha = 0.5, position="identity")+
    facet_grid(stim.race ~ stim.gen) +
    #theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          #legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12, face="bold"),
          legend.text=element_text(size=12)) +
          scale_color_colorblind() +
          labs(y = "Density", x = "Score",
          fill = "Hair (Unbalanced)") 

 

ggsave("results/figures/atypical_all_50face_unbalanced_hist.svg", plot=race_hair_unbalanced, width=14)
ggsave(filename="results/figures/atypical_all_50face_unbalanced_hist.jpg", plot=race_hair_unbalanced,width=14)





race_hair<- ggplot(gan_hair, aes(x=discriminator_score, fill = hairlength)) +
    geom_density(alpha = 0.25)+
    facet_grid(stim.race ~ stim.gen) +
    #theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          #legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12, face="bold"),
          #legend.title=element_blank(),
          legend.text=element_text(size=12)) +
        scale_color_colorblind() +
  labs(y = "Density", x = "Score",
       fill = "Hair") 


 


ggsave("results/figures/atypical_all_50face_dist.svg", plot=race_hair, width=14)
ggsave("results/figures/atypical_all_50face_dist.jpg", plot=race_hair, width=14)


plot_hair_unbalanced<- ggplot(gan_hair_unbalanced, aes(x=discriminator_score, fill = hairlength)) +
    geom_density(alpha = 0.25) +
    #theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12, face="bold"),
          #legend.title=element_blank(),
          legend.text=element_text(size=12)) +
        scale_color_colorblind() +
  labs(y = "Density", x = "Score",
       fill = "Hair (Unbalanced)") 


ggsave("results/figures/atypical_all_50face_dist_aggregate.svg", plot=plot_hair_unbalanced)
ggsave("results/figures/atypical_all_50face_dist_aggregate.jpg", plot=plot_hair_unbalanced)




plot_hair<- ggplot(gan_hair, aes(x=discriminator_score, fill = hairlength)) +
  geom_histogram(alpha = 0.5, position="identity")+
  facet_grid(stim.race ~ stim.gen)+
  #theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          legend.position = "bottom",
          axis.text=element_text(size=12),
        axis.title=element_text(size=14),
        legend.title=element_text(size=12),
        legend.text=element_text(size=12)) +
        scale_color_colorblind() +
  labs(y = "Count", x = "Score",
       fill = "Hair") 



plot_hair
ggsave("results/figures/atypical_all_50faces_hist.svg", plot=plot_hair)
ggsave("results/figures/atypical_all_50faces_hist.jpg", plot=plot_hair)


plot_race<- ggplot(gan_hair, aes(x=discriminator_score, fill = stim.race)) +
    geom_density(alpha = 0.25) +
    #theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12),
          #legend.title=element_blank(),
          legend.text=element_text(size=12)) +
        scale_color_colorblind() +
  labs(y = "Density", x = "Score",
       fill = "Race") 


ggsave("results/figures/atypical_all_50face_race_dist_aggregate.svg", plot=plot_race)
ggsave("results/figures/atypical_all_face_race_dist_aggregate.jpg", plot=plot_race)




plot_hair_raincloud <- ggplot(gan_hair, aes(hairlength, discriminator_score)) + 
  ggdist::stat_halfeye(adjust = .5, width = .3, .width = c(0.5, 1), justification = -.3) + 
  geom_boxplot(width = .1, outlier.shape = NA) +
  ggdist::stat_dots(side = "left", dotsize = .4, justification = 1.1, binwidth = 5)+
  facet_wrap(~stim.race)

plot_hair_raincloud
ggsave("results/figures/atypical_all_50face_raincloud.svg", plot=plot_hair_raincloud)
ggsave("results/figures/atypical_all_50face_raincloud.jpg", plot=plot_hair_raincloud)

###### predicting score ~ race #######

## mod_race <- lm(discriminator_score~stim.race,data=gan_hair)
## summary(mod_race)
## anova(mod_race)

## mod_race_contrlum <- lm(discriminator_score~stim.race+luminance,data=gan_hair)
## summary(mod_race_contrlum)
## anova(mod_race)


## mod_racehair <- lm(discriminator_score~race*hairlength,data=gan_hair)
## summary(mod_racehair)
## anova(mod_racehair)


## mod_racehair_contrlum <- lm(discriminator_score~stim.race*hairlength+luminance,
##                             data=gan_hair)
## summary(mod_racehair_contrlum)
## anova(mod_racehair_contrlum)
