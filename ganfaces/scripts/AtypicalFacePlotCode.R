lib<-c("lme4","readr","tidyr","effects","ggplot2","psych","MASS",
       "Rmisc","plyr","dplyr","lmerTest","ggthemes","lsmeans",
       "pastecs","sjstats","car","readxl","ggdist", "svglite","gghdr")
lapply(lib,require,character.only=TRUE)

gan_hair<-read.csv("data/AtypicalFaceData.csv")

mean_gan_hair<- summarySE(data=gan_hair, measurevar = "discriminator_score",
                          groupvars = c("race","hair_length"),
                          na.rm = FALSE, conf.interval = 0.95)
                          
race_hair<- ggplot(gan_hair, aes(x=discriminator_score, fill = hair_length)) +
    geom_density(alpha = 0.25)+
    facet_grid(race ~ .) +
    theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12),
          #legend.title=element_blank(),
          legend.text=element_text(size=12)) +
        scale_color_colorblind() +
  labs(y = "Density", x = "Score",
       fill = "Hair") 


 


ggsave("results/figures/atypical_mens_face_dist.svg", plot=race_hair)
ggsave("results/figures/atypical_mens_face_dist.jpg", plot=race_hair)


plot_hair<- ggplot(gan_hair, aes(x=discriminator_score, fill = hair_length)) +
    geom_density(alpha = 0.25) +
    theme_minimal() +
    theme(strip.text.y = element_text(size = 14),
          legend.position="bottom",
          axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.title=element_text(size=12),
          #legend.title=element_blank(),
          legend.text=element_text(size=12)) +
        scale_color_colorblind() +
  labs(y = "Density", x = "Score",
       fill = "Hair") 


ggsave("results/figures/atypical_mens_face_dist_aggregate.svg", plot=plot_hair)
ggsave("results/figures/atypical_mens_face_dist_aggregate.jpg", plot=plot_hair)




plot_hair<- ggplot(gan_hair, aes(x=discriminator_score, fill = hair_length)) +
  geom_histogram(alpha = 0.5)+
  facet_grid(race ~ .)+
  theme_minimal() +
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
ggsave("results/figures/atypical_mens_face_hist.svg", plot=plot_hair)
ggsave("results/figures/atypical_mens_face_hist.jpg", plot=plot_hair)


plot_race<- ggplot(gan_hair, aes(x=discriminator_score, fill = race)) +
    geom_density(alpha = 0.25) +
    theme_minimal() +
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


ggsave("results/figures/atypical_mens_face_race_dist_aggregate.svg", plot=plot_race)
ggsave("results/figures/atypical_mens_face_race_dist_aggregate.jpg", plot=plot_race)




plot_hair_raincloud <- ggplot(gan_hair, aes(hair_length, discriminator_score)) + 
  ggdist::stat_halfeye(adjust = .5, width = .3, .width = c(0.5, 1), justification = -.3) + 
  geom_boxplot(width = .1, outlier.shape = NA) +
  ggdist::stat_dots(side = "left", dotsize = .4, justification = 1.1, binwidth = 5)+
  facet_wrap(~race)

plot_hair_raincloud
ggsave("results/figures/atypical_mens_face_raincloud.svg", plot=plot_hair_raincloud)
ggsave("results/figures/atypical_mens_face_raincloud.jpg", plot=plot_hair_raincloud)

###### predicting score ~ race #######

mod_race <- lm(discriminator_score~race,data=gan_hair)
summary(mod_race)
anova(mod_race)

mod_race_contrlum <- lm(discriminator_score~race+luminance,data=gan_hair)
summary(mod_race_contrlum)
anova(mod_race)


mod_racehair <- lm(discriminator_score~race*hair_length,data=gan_hair)
summary(mod_racehair)
anova(mod_racehair)


mod_racehair_contrlum <- lm(discriminator_score~race*hair_length+luminance,
                            data=gan_hair)
summary(mod_racehair_contrlum)
anova(mod_racehair_contrlum)
