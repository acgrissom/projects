rm(list=ls())
lib<-c("lme4","readr","tidyr","effects","ggplot2","psych","MASS",
       "Rmisc","plyr","dplyr","lmerTest","ggthemes","lsmeans",
       "pastecs","sjstats","car","readxl","ggdist","nnet")
lapply(lib,require,character.only=TRUE)

## read in csv for racial classification of top and bottom 100 faces
gan_ffhq<-read.csv(file.choose())

gan_ffhq_perc<-gan_ffhq %>% count(race, top.bot) %>%    # Group by race and top or bottom, then count number in each group
         mutate(pct=n/sum(n/2),               # Calculate percent - divided n by two since each group has ~100
                ypos = cumsum(n) - 0.5*n)  # Calculate cumulative percentage

## rename levels
levels(gan_ffhq_perc$top.bot)[levels(gan_ffhq_perc$top.bot)=="top"] <- "Top 100"
levels(gan_ffhq_perc$top.bot)[levels(gan_ffhq_perc$top.bot)=="bot"] <- "Bottom 100" 

### plot code
gan_ffhq_plot<-ggplot(gan_ffhq_perc, aes(fill=race, y = n, x=top.bot)) + 
  geom_bar(position="stack", stat="identity")+
  theme_few() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12),
        legend.title=element_text(size=12),
        legend.text=element_text(size=12)) +
  scale_color_economist() +
  labs(y = "Count", x = "Top or Bottom 100 Faces",
       fill = "Rated Race of Image") 

gan_ffhq_plot

### running multinomial regression
gan_ffhq$race <- relevel(gan_ffhq$race, ref = "White")
mod_main <- multinom(race ~ top.bot, data = gan_ffhq)
summary(mod_main)

##calculate z-values
z <- summary(mod_main)$coefficients/summary(mod_main)$standard.errors
z

##calculate p-values
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p

