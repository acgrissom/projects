import pymc
from pymc import Normal, HalfCauchy, sample
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

"""
coords and dims: https://oriolabrilpla.cat/en/blog/posts/2020/pymc3-arviz.html
model comparison: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html
"""
samples = 10000
if __name__ == "__main__":
    df = pd.read_csv('data/AtypicalFaceData.csv')
    #dummies_df = pd.get_dummies(df)
    race_names = df.race.unique()
    race_names = df.hair_length.unique()
    race_idx, races = pd.factorize(df.race)
    hair_idx, hair_lengths = pd.factorize(df.hair_length)
    coords = {
        "race": race_names,
        "hair": hair_lengths,
        "param": ["x_black"],
        "obs_id": np.arange(len(df))
    }

    print(race_names)


    
    
    
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
    x_race = races.values
    x_hair = hair_lengths.values
    #long_std = x_long.std()
    #short_std = x_short.std()
    

    
    score_std = df.discriminator_score.std()
    asian_mean_score  = df[df['race'] == 'Asian'].discriminator_score.mean()
    black_mean_score  = df[df['race'] == 'Black'].discriminator_score.mean()
    white_mean_score  = df[df['race'] == 'White'].discriminator_score.mean()
    
    with pymc.Model(coords=coords) as hier_reg:  # model specifications in PyMC are wrapped in a with-statement
        # constant data
        hair_length = pymc.ConstantData("hair_length", hair_idx)
        
        # global model params
                #race_idx = pymc.Data("race_idx", race_code, dims="race")
        #hair_idx = pymc.Data("hair_idx", hair_idxs, dims="hair")
        beta_race = pymc.Normal("beta_race", 0, sigma=score_std, dims="hair")
        # Define priors


        
        beta_red = Normal("red", 0, sigma=red_std)
        beta_green = Normal("green", 0, sigma=green_std)
        beta_blue = Normal("blue", 0, sigma=blue_std)


        
        sigma = HalfCauchy("sigma", beta=score_std)
        y =  beta_race[hair_idx] * race_idx # +  beta_hair[hair_idx] * x_hair # + beta_red * xr + beta_green * xg + beta_blue * xb
        #beta_black * x_black + beta_asian * x_asian + beta_white * x_white +
        #beta_long * x_long + beta_short * x_short +
        
        likelihood = Normal("y", mu=y,
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

        


    
    compare_loo = az.compare(
        {"hier_reg": hier_trace, "lin_reg": reg_trace}, ic="loo"
    )
    print(compare_loo)
    az.plot_compare(compare_loo, insample_dev=False);
    plt.show()
