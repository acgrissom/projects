import pymc
from pymc import Normal, HalfCauchy, sample
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt


"""
https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html
"""
samples = 10000
if __name__ == "__main__":
    df = pd.read_csv('data/AtypicalFaceData.csv')
    dummies_df = pd.get_dummies(df)
    long_mean_score = df[df['hair_length'] == 'long'].discriminator_score.mean()
    short_mean_score = df[df['hair_length'] == 'short'].discriminator_score.mean()
    score_mean = df.discriminator_score.mean()
    red_std  = df['red_mean'].std()
    blue_std  = df['blue_mean'].std()
    green_std  = df['green_mean'].std()
    
    asian_std  = df[df['race'] == 'Asian'].discriminator_score.std()
    black_std  = df[df['race'] == 'Black'].discriminator_score.std()
    white_std  = df[df['race'] == 'White'].discriminator_score.std()
    xr = df.red_mean.values
    xg = df.green_mean.values
    xb = df.blue_mean.values
    x_asian = dummies_df.race_Asian.values
    x_black = dummies_df.race_Black.values
    x_white = dummies_df.race_White.values
    x_long = dummies_df.hair_length_long.values
    x_short = dummies_df.hair_length_short.values

    long_std = x_long.std()
    short_std = x_short.std()
    

    
    score_std = df.discriminator_score.std()
    asian_mean_score  = df[df['race'] == 'Asian'].discriminator_score.mean()
    black_mean_score  = df[df['race'] == 'Black'].discriminator_score.mean()
    white_mean_score  = df[df['race'] == 'White'].discriminator_score.mean()
    
    with pymc.Model() as hier_reg:  # model specifications in PyMC are wrapped in a with-statement
    
        # Define priors


        asian_mu = Normal("Asian_sigma", 0, sigma=asian_std)
        black_mu = Normal("Black_sigma", 0, sigma=black_std)
        white_mu = Normal("White_sigma", 0, sigma=white_std)
        long_mu = Normal("long_sigma", 0, sigma=long_std)
        short_mu = Normal("short_sigma", 0, sigma=short_std)
        
        beta_long = Normal("long", 0, sigma=long_std)
        beta_short = Normal("short", 0, sigma=short_std)

        
        beta_red = Normal("red", 0, sigma=red_std)
        beta_green = Normal("green", 0, sigma=green_std)
        beta_blue = Normal("blue", 0, sigma=blue_std)


        # black_short_std  = df[(df['race'] == 'Black') &
        # (df['hair_length'] == 'short')].discriminator_score.std()

        # black_long_std  = df[(df['race'] == 'Black') &
        #                             (df['hair_length'] == 'long')].discriminator_score.std()

        # white_short_std  = df[(df['race'] == 'White') &
        # (df['hair_length'] == 'short')].discriminator_score.std()

        # white_long_std  = df[(df['race'] == 'White') &
        #                             (df['hair_length'] == 'long')].discriminator_score.std()


        # asian_short_std  = df[(df['race'] == 'Asian') &
        # (df['hair_length'] == 'short')].discriminator_score.std()

        # asian_long_std  = df[(df['race'] == 'Asian') &
        #                             (df['hair_length'] == 'long')].discriminator_score.std()


        

        
        sigma = HalfCauchy("sigma", beta=score_std)
        intercept = Normal("Intercept", score_mean, sigma=score_std)
        beta_asian = Normal("Asian", asian_mu, sigma=asian_std)
        beta_black = Normal("Black", black_mu, sigma=black_std)
        beta_white = Normal("White", white_mu, sigma=white_std)
        # Define likelihood
        # likelihood = Normal("y", mu=intercept + beta_red * xr +
        #                     beta_green * xg + beta_blue * xb +
        #                     beta_black * x_black + beta_asian *
        #                     x_asian + beta_white * x_white + beta_long
        #                     * x_long + beta_short * x_short,
        #                     sigma=sigma,
        #                     observed=df.discriminator_score.values)

        likelihood = Normal("y", mu=intercept +
                            beta_red * xr +
                            beta_green * xg + beta_blue * xb +
                            beta_black * x_black +
                            beta_asian * x_asian + beta_white * x_white +
                            beta_long * x_long + beta_short * x_short +
                            0,
                            sigma=sigma,
                            observed=df.discriminator_score.values)

        
        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        hier_trace = sample(samples, return_inferencedata=True, idata_kwargs={"log_likelihood": True})
        var_names=["red", "green", "blue", "Black", "White","Asian", "long", "short"]
        #var_names=["Black", "White","Asian", "long", "short"]
        az.plot_trace(hier_trace, var_names=var_names);
        #plt.show()
        hier_ppc = pymc.sample_posterior_predictive(hier_trace, extend_inferencedata=True)
        print(hier_ppc)
        print(az.loo(hier_trace, hier_reg))
        az.plot_ppc(hier_trace, num_pp_samples=100);
        plt.show()

        

    with pymc.Model() as lin_reg:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10)
        intercept = Normal("Intercept", 0, sigma=score_std)
        beta_red = Normal("red", 0, sigma=red_std)
        beta_green = Normal("green", 0, sigma=green_std)
        beta_blue = Normal("blue", 0, sigma=blue_std)
        beta_asian = Normal("Asian", 0, sigma=asian_std)
        beta_black = Normal("Black", 0, sigma=black_std)
        beta_white = Normal("White", 0, sigma=white_std)
        beta_long = Normal("long", 0, sigma=long_std)
        beta_short = Normal("short", 0, sigma=short_std)
        xr = df.red_mean.values
        xg = df.green_mean.values
        xb = df.blue_mean.values
        x_asian = dummies_df.race_Asian.values
        x_black = dummies_df.race_Black.values
        x_white = dummies_df.race_White.values
        x_long = dummies_df.hair_length_long.values
        x_short = dummies_df.hair_length_short.values
        
        # Define likelihood
        likelihood = Normal("y", mu=intercept +
                            beta_red * xr +
                            beta_green * xg + beta_blue * xb +
                            beta_black * x_black + beta_asian *
                            x_asian + beta_white * x_white + beta_long
                            * x_long + beta_short * x_short,
                            sigma=sigma,
                            observed=df.discriminator_score.values)
        
        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        reg_trace = sample(samples, return_inferencedata=True,
                           idata_kwargs={"log_likelihood": True})
        var_names=["red", "green", "blue", "Black", "White","Asian", "long", "short"]
        #var_names=["Black", "White","Asian", "long", "short"]
        az.plot_trace(reg_trace, var_names=var_names);
        plt.show()
        print(az.loo(reg_trace, lin_reg))

    
    compare_loo = az.compare(
        {"hier_reg": hier_trace, "lin_reg": reg_trace}, ic="loo"
    )
    print(compare_loo)
    az.plot_compare(compare_loo, insample_dev=False);
    plt.show()
