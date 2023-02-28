# https://rpubs.com/Qsheep/BayesianLinearRegression
library(mlbench)
library(rstanarm)
library(bayestestR)
library(bayesplot)
library(insight)
library(broom)
library(ggplot2)
library(ggdensity)
library(rstanarm)
library(bayesplot)
library(fastDummies)
csv = read.csv("data/AtypicalFaceData.csv", header=TRUE)
pars = c("luminance", "red_mean", "green_mean", "blue_mean", "race", "hair_length", "discriminator_score")
pars_norgb = c("race", "hair_length", "discriminator_score")

df = csv[c("luminance", "red_mean", "green_mean", "blue_mean", "race", "hair_length", "discriminator_score")]
df$luminance = NULL


asian_df <- ifelse(df$race == "Asian", 1, 0)
black_df <- ifelse(df$race == "Black", 1, 0)
white_df <- ifelse(df$race == "White", 1, 0)
hair_long_df <- ifelse(df$hair == "long", 1, 0)
hair_short_df <- ifelse(df$hair == "short", 1, 0)
## red_norm_df <- df$red_mean / 255.0
## green_norm_df <- df$green_mean / 255.0
## blue_norm_df <- df$blue_mean / 255.0
red_norm_df <- df$red_mean 
green_norm_df <- df$green_mean 
blue_norm_df <- df$blue_mean 


cat_df <- data.frame(
    discriminator_score = df$discriminator_score,
                                        #red_mean = df$red_mean,
                                        #green_mean =df$green_mean,
                                        #blue_mean = df$blue_mean,
    red_mean = red_norm_df,
    green_mean = green_norm_df,
    blue_mean = blue_norm_df,
    race_Asian = asian_df,
    race_Black = black_df,
    race_White = white_df,
    hair_long = hair_long_df,
    hair_short = hair_short_df)
#df <- subset(df, select = -c(race_Asian, race_Black, race_White, hair_long, hair_short))

#cat_df = csv[pars_norgb]


#df <- model.matrix(~ race, data = df)
print(head(df))
#prior = laplace(location = 0, scale = NULL, autoscale = FALSE)
#prior = normal(location = 0, scale = NULL, autoscale = FALSE)
 prior = student_t(df = 1, location = 0, scale = NULL, autoscale = FALSE)

pars = c("red_mean", "green_mean", "blue_mean","race_Black", "race_White", "race_Asian", "hair_long", "hair_short")
pars_norgb = c("race_White", "race_Asian", "race_Black","hair_long", "hair_short")
pars_rgb_only = c("red_mean", "green_mean", "blue_mean")

pars = pars_norgb


cat_df = cat_df[c("discriminator_score", pars)]
cat_df = df
model_bayes <- stan_glm(discriminator_score~., data=cat_df, cores=8, iter=100000, prior=prior)
print(model_bayes)
## mcmc_dens(model_bayes)
## hdi(model_bayes)
options(mc.cores=parallel::detectCores()) 
posterior <- as.matrix(model_bayes)
plot_title <- ggtitle("Posterior distributions",
                      "with medians and 95% HDI")

mcmc_areas(posterior,
           pars = pars,
           prob = 0.95) + plot_title
ggsave('bayes_dens.png')
ggsave('bayes_dens.svg')
fit = model_bayes
yrep = posterior_predict(fit, draws = 500)
ppc_dens_overlay(y = fit$y, yrep = yrep)
ggsave('bayes_ppc.svg')
ggsave('bayes_ppc.png')

ppc_stat_grouped(y = fit$y,
                 yrep = yrep,
                 group = cat_df$race_Black,
                 stat = "median")

ggsave('bayes_ppc_black.svg')
ggsave('bayes_ppc_black.png')


ppc_stat_grouped(y = fit$y,
                 yrep = yrep,
                 group = cat_df$race_Asian,
                 stat = "median")

ggsave('bayes_ppc_asian.svg')
ggsave('bayes_ppc_asian.png')


ppc_stat_grouped(y = fit$y,
                 yrep = yrep,
                 group = cat_df$race_White,
                 stat = "median")

ggsave('bayes_ppc_white.svg')
ggsave('bayes_ppc_white.png')


