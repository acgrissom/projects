import pymc
from pymc import Normal, HalfNormal, sample, HalfStudentT, HalfCauchy
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
az.style.use("arviz-darkgrid")

    

if __name__ == "__main__":
    #df = pd.read_csv('data/AtypicalFaceData.csv')
    df = pd.read_csv("data/cleaned/Study2_GoogleFaces_full_unbalanced.csv")
    dummies_df = pd.get_dummies(df)

    red_std  = df['red_mean'].std()
    blue_std  = df['blue_mean'].std()
    green_std  = df['green_mean'].std()
    score_mean = df.discriminator_score.mean()
    score_std = df.discriminator_score.std()
    #asian_std  = df[df['race'] == 'Asian'].discriminator_score.std()
    #black_std  = df[df['race'] == 'Black'].discriminator_score.std()
    #white_std  = df[df['race'] == 'White'].discriminator_score.std()


    with pymc.Model() as regression:  # model specifications in PyMC are wrapped in a with-statement
        gamma = 2
        # Define priors

        sigma_red = HalfStudentT("sigma_red",sigma=1, nu=10)
        sigma_green = HalfStudentT("sigma_green",sigma=1, nu=10)
        sigma_blue = HalfStudentT("sigma_blue",sigma=1, nu=10)
        sigma_int = HalfStudentT("sigma_int",sigma=1, nu=10)
        #mu_int = HalfNormal("mu_score", mu=score_mean, sigma = 1)
        
        sigma = HalfStudentT("sigma_score", sigma=1, nu = 10)
        intercept = Normal("$\\beta_0$", score_mean, sigma_int)
        beta_red = Normal("red", 0, sigma=sigma_red)
        beta_green = Normal("green", 0, sigma=sigma_green)
        beta_blue = Normal("blue", 0, sigma=sigma_blue)
        # beta_asian = Normal("Asian", 0, sigma=20)
        # beta_black = Normal("Black", 0, sigma=20)
        # beta_white = Normal("White", 0, sigma=20)
        # beta_long = Normal("long", 0, sigma=20)
        # beta_short = Normal("short", 0, sigma=20)
        xr = df.red_mean.values
        xg = df.green_mean.values
        xb = df.blue_mean.values
        # x_asian = dummies_df.race_Asian.values
        # x_black = dummies_df.race_Black.values
        # x_white = dummies_df.race_White.values
        # x_long = dummies_df.hair_length_long.values
        # x_short = dummies_df.hair_length_short.values
        
        # Define likelihood
        likelihood = Normal("y", mu=intercept +
                            beta_red * xr +  beta_green * xg + beta_blue * xb,# +
                            #beta_black * x_black + beta_asian *
                            #x_asian + beta_white * x_white + beta_long
                            #* x_long + beta_short * x_short,
                            sigma=sigma,
                            observed=df.discriminator_score.values)
        
        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        trained_reg = sample(50000,
                             chains=10,
                             cores=8,
                             return_inferencedata=True)
        var_names=["red", "green", "blue", "Black", "White","Asian", "long", "short"]
        var_names=["Black", "White","Asian", "long", "short"]
        var_names=["red", "green", "blue", "$\\beta_0$"]
        az.plot_trace(trained_reg,
                      var_names=var_names,
                      #figsize=(7,3.5),
                      kind="rank_bars",
                      compact=True,
                      combined=True,
                      rug=False)
        plt.savefig('results/figures/rgb_posteriors.svg')
        plt.savefig('results/figures/rgb_posteriors.jpg')

        post_plot = az.plot_posterior(trained_reg,var_names=var_names,
                                      hdi_prob=0.95,
                                      figsize=(7,7),
                                      textsize=10,
                                      grid=(2,2)
                                      )
        plt.savefig('results/figures/rgb_posteriors_hdi.svg')
        plt.savefig('results/figures/rgb_posteriors_hdi.jpg')
        plt.show()

        #ax, = pymc.plot_posterior(trained_reg)
        # ax, = pymc.plot_posterior(trained_reg,
        #                           hdi_prob=0.95,
        #                           figsize=(7,7),
        #                           kind="rank_bars")
        plt.show()
        
