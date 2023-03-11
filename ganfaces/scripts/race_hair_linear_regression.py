import pymc
from pymc import Normal, HalfNormal, sample, StudentT, Gamma
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
az.style.use("arviz-darkgrid")

    

if __name__ == "__main__":
    #df = pd.read_csv('data/AtypicalFaceData.csv')
    df = pd.read_csv("data/cleaned/Study2_GoogleFaces_full_unbalanced.csv")
    dummies_df = pd.get_dummies(df)
    print(dummies_df['stim.race_Asian'])
    red_std  = df['red_mean'].std()
    blue_std  = df['blue_mean'].std()
    green_std  = df['green_mean'].std()
    score_mean = df.discriminator_score.mean()
    score_std = df.discriminator_score.std()
    asian_std  = df[df['stim.race'] == 'Asian'].discriminator_score.std()
    black_std  = df[df['stim.race'] == 'Black'].discriminator_score.std()
    white_std  = df[df['stim.race'] == 'White'].discriminator_score.std()
    man_std = df[df['stim.gen'] == 'men'].discriminator_score.std()
    woman_std = df[df['stim.gen'] == 'women'].discriminator_score.std()
    long_std = df[df['hairlength'] == 'long'].discriminator_score.std()
    short_std = df[df['hairlength'] == 'short'].discriminator_score.std()

    with pymc.Model() as regression:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        # nu = 1
        # nu_black = nu
        # nu_white = nu
        # nu_asian = nu
        # nu_long = nu
        # nu_short = nu
        # nu_man = nu
        # nu_woman = nu

        nu_black = HalfNormal("nu_black",10) 
        nu_white = HalfNormal("nu_white",10) 
        nu_asian = HalfNormal("nu_asian",10)
        nu_long = HalfNormal("nu_long",10) 
        nu_short = HalfNormal("nu_short",10)
        nu_man = HalfNormal("nu_man",10)
        nu_woman = HalfNormal("nu_woman",10)
        sigma = HalfNormal("sigma", sigma=score_std)
        intercept = Normal("$\\beta_0$", score_mean, score_std)
        beta_red = Normal("red", 0, sigma=red_std)
        beta_green = Normal("green", 0, sigma=green_std)
        beta_blue = Normal("blue", mu=0, sigma=blue_std)
        
        # beta_asian = pymc.Normal("Asian", mu=0, sigma=asian_std)
        # beta_black = pymc.Normal("Black", mu=0, sigma=black_std)
        # beta_white = pymc.Normal("White", mu=0, sigma=white_std)
        # beta_long = pymc.Normal("long", mu=0, sigma=long_std)
        # beta_short = pymc.Normal("short",mu=0, sigma=short_std)
        # beta_man = pymc.Normal("man", sigma=man_std)
        # beta_woman = pymc.Normal("woman", sigma=woman_std)

        
        beta_asian = pymc.StudentT("Asian",nu=nu_asian, mu=0, sigma=asian_std)
        beta_black = pymc.StudentT("Black",nu=nu_black, mu=0, sigma=black_std)
        beta_white = pymc.StudentT("White",nu=nu_white, mu=0, sigma=white_std)
        beta_long = pymc.StudentT("long", nu=nu_long,mu=0, sigma=long_std)
        beta_short = pymc.StudentT("short", nu=nu_short,mu=0, sigma=short_std)
        beta_man = pymc.StudentT("man", nu=nu_man,mu=0, sigma=man_std)
        beta_woman = pymc.StudentT("woman",nu=nu_woman,mu=0, sigma=woman_std)

        xr = df.red_mean.values
        xg = df.green_mean.values
        xb = df.blue_mean.values
        x_asian = dummies_df['stim.race_Asian'].values
        x_black = dummies_df['stim.race_Black'].values
        x_white = dummies_df['stim.race_White'].values
        x_man = dummies_df['stim.gen_men'].values
        x_woman = dummies_df['stim.gen_women'].values

        x_long = dummies_df.hairlength_long.values
        x_short = dummies_df.hairlength_short.values

        
        # Define likelihood
        likelihood = Normal("y", mu=intercept +
                            #beta_red * xr +  beta_green * xg + beta_blue * xb +
                            beta_black * x_black +
                            beta_asian * x_asian +
                            beta_white * x_white +
                            beta_long * x_long + beta_short * x_short +
                            beta_man * x_man + beta_woman * x_woman,
                            sigma=sigma,
                            observed=df.discriminator_score.values)
        
        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        trained_reg = sample(10000, return_inferencedata=True)
        var_names=["Black", "White","Asian", "long", "short", "man", "woman", "$\\beta_0$"]
        #var_names=["red", "green", "blue", "Black", "White","Asian", "long", "short"]        #var_names=["Black", "White","Asian", "long", "short"]
        #var_names=["red", "green", "blue"]
        az.plot_trace(trained_reg,
                      var_names=var_names,
                      figsize=(7,7),
                      #kind="rank_bars",
                      compact=True,
                      combined=True,
                      rug=False);
        plt.savefig('results/figures/race_hair_trace.svg')
        plt.savefig('results/figures/race_hair_trace.jpg')
        post_plot = az.plot_posterior(trained_reg,var_names=var_names,
                                      hdi_prob=0.95,
                                      figsize=(7,7),
                                      textsize=10,
                                      grid=(4,2))
                                      
        # ax, = pymc.plot_posterior(trained_reg,
        #                           hdi_prob=0.95,
        #                           figsize=(7,7),
        #                           kind="rank_bars")
        plt.savefig('results/figures/race_hair_posteriors.svg')
        plt.savefig('results/figures/race_hair_posteriors.jpg')
        plt.show()

