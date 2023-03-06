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
csv = read.csv("data/correct_LAB_format_images_data.csv", header=TRUE)
df = csv[c("luminance", "red_mean", "green_mean", "blue_mean", "scores_rgb")]
df$luminance = NULL
head(df)
cores = parallel::detectCores()
options(mc.cores=cores) 
model_bayes <- stan_glm(scores_rgb~., data=df, seed=111, cores=cores, iter=20000)
## mcmc_dens(model_bayes)
## hdi(model_bayes)

posterior <- as.matrix(model_bayes)
plot_title <- ggtitle("Posterior distributions",
                      "with medians and 95% HDI")
mcmc_areas(posterior,
           pars = c("red_mean", "green_mean", "blue_mean"),
           prob = 0.95) + plot_title
ggsave('bayes_dens.png')


