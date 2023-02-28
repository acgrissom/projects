import pymc
from pymc import Normal, HalfCauchy, sample
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt


    

if __name__ == "__main__":
    df = pd.read_csv('data/AtypicalFaceData.csv')
    dummies_df = pd.get_dummies(df)
    with pymc.Model() as regression:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10)
        intercept = Normal("Intercept", 0, sigma=20)
        beta_red = Normal("red", 0, sigma=20)
        beta_green = Normal("green", 0, sigma=20)
        beta_blue = Normal("blue", 0, sigma=20)
        beta_asian = Normal("Asian", 0, sigma=20)
        beta_black = Normal("Black", 0, sigma=20)
        beta_white = Normal("White", 0, sigma=20)
        beta_long = Normal("long", 0, sigma=20)
        beta_short = Normal("short", 0, sigma=20)
        xr = df.red_mean.values
        xg = df.green_mean.values
        xb = df.blue_mean.values
        x_asian = dummies_df.race_Asian.values
        x_black = dummies_df.race_Black.values
        x_white = dummies_df.race_White.values
        x_long = dummies_df.hair_length_long.values
        x_short = dummies_df.hair_length_short.values
        
        # Define likelihood
        likelihood = Normal("y", mu=intercept + beta_red * xr +
                            beta_green * xg + beta_blue * xb +
                            beta_black * x_black + beta_asian *
                            x_asian + beta_white * x_white + beta_long
                            * x_long + beta_short * x_short,
                            sigma=sigma,
                            observed=df.discriminator_score.values)
        
        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        trained_reg = sample(10000, return_inferencedata=True)
        az.plot_trace(trained_reg, var_names=["red", "green", "blue", "Black", "White","Asian", "long", "short"]);
        plt.show()
        
