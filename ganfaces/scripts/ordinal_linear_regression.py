import pymc
from pymc import Normal, HalfNormal, sample, StudentT, Gamma, HalfCauchy, HalfStudentT
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from pandasql import sqldf
from collections import defaultdict
az.style.use("arviz-darkgrid")

categories = ["Masculinity",
              "Femininity",
              "Skintone",
              "Afrocentric",
              "Eurocentric",
              "Asiocentric",
              "Hair"]


def get_ordinal_values(df):
    x_columns_dict = defaultdict(list)
    for cat in categories:
        std_str = cat + "Std"
        mean_str = cat + "Mean"
        x_columns_dict[mean_str] = df[mean_str].values
        x_columns_dict[std_str] = df[std_str].values
    new_df = pd.DataFrame(x_columns_dict)
    print(new_df.head())
    return new_df
        

if __name__ == "__main__":
    #df = pd.read_csv('data/cleaned/Study2_50_aggregated.csv')
    df=pd.read_csv('data/cleaned/Study2_unbalanced_aggregated.csv')
#    df = pd.read_csv("data/cleaned/Study2_GoogleFaces_full_unbalanced.csv")
#    print(df['Asiocentric'].values)
    ord_vals = get_ordinal_values(df)
    dummies_df = pd.get_dummies(df)
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

    asio_mean =df["AsiocentricMean"].mean()
    afro_mean = df["AfrocentricMean"].mean()
    euro_mean=df["EurocentricMean"].mean()
    hair_mean=df["HairMean"].mean()
    skin_mean=df["SkintoneMean"].mean()
    masc_mean =df["MasculinityMean"].mean()
    fem_mean=df["FemininityMean"].mean()

    
    

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

        gamma = 2
        nu = HalfCauchy("nu",gamma) 
        nu_asio = nu#HalfCauchy("nu_asio",gamma) 
        nu_afro = nu# HalfCauchy("nu_afro",gamma) 
        nu_euro = nu#HalfCauchy("nu_euro",gamma) 
        nu_hair = nu#HalfCauchy("nu_hair",gamma) 
        nu_skin = nu#HalfCauchy("nu_skin",gamma)
        nu_masc = nu#HalfCauchy("nu_mas",gamma)
        nu_fem = nu#HalfCauchy("nu_fem",gamma)
        nu_int = nu#HalfCauchy("nu_int",gamma)

        sigma_asio = HalfStudentT("sigma_asio",sigma=1, nu=10)
        sigma_afro = HalfStudentT("sigma_afro",sigma=1, nu=10)
        sigma_euro = HalfStudentT("sigma_euro",sigma=1, nu=10)
        sigma_hair = HalfStudentT("sigma_hair",sigma=1, nu=10) 
        sigma_skin = HalfStudentT("sigma_skin",sigma=1, nu=10)
        sigma_masc = HalfStudentT("sigma_masc",sigma=1, nu=10)
        sigma_fem = sigma_masc
        sigma_int = HalfStudentT("sigma_int",sigma=1, nu=10)
        sigma_score = HalfStudentT("sigma_score",sigma=1, nu=10)



        mu_asio = Normal("mu_asio",mu=0,sigma=sigma_asio) 
        mu_afro = Normal("mu_afro",mu=0, sigma=sigma_afro) 
        mu_euro = Normal("mu_euro",mu=0,sigma=sigma_euro) 
        mu_hair = Normal("mu_hair",mu=0, sigma=sigma_hair) 
        mu_skin = Normal("mu_skin",mu=0,sigma=sigma_skin)
        mu_masc = Normal("mu_masc",mu=0, sigma=sigma_masc)
        mu_fem = Normal("mu_fem",mu=0,sigma=sigma_fem)
        mu_int = Normal("mu_int",mu=score_mean,sigma=sigma_int)
        
        
        sigma = HalfCauchy("sigma", beta=10)
        
        
        beta_asio = StudentT("Asiocentric",
                             nu=nu_asio,
                             #mu=df["AsiocentricMean"].mean(),
                             mu=mu_asio,
                             sigma=sigma_asio)
        
        beta_afro = StudentT("Afrocentric",
                             nu=nu_afro,
                             #mu=df["AfrocentricMean"].mean(),
                             mu=mu_afro,
                             sigma=sigma_afro)
        
        
        beta_euro = StudentT("Eurocentric",
                             nu=nu_euro,
                             #mu=df["EurocentricMean"].mean(),
                             mu=mu_euro,
                             sigma=sigma_euro)
        
        beta_skin = StudentT("Skintone",
                             nu=nu_skin,
                             mu=mu_skin,
                             sigma=sigma_skin)
        
        beta_masc = StudentT("Masculinity",
                             nu=nu_masc,
                             mu=mu_masc,
                             sigma=sigma_masc)
        
        
        beta_fem =  StudentT("Femininity",
                             nu=nu_fem,
                             mu=mu_fem,
                             sigma=sigma_fem)
        
        
        beta_hair = StudentT("Hair",
                             nu=nu_hair,
                             mu=mu_hair,
                             sigma=sigma_hair)
        
        intercept = StudentT("$\\beta_0$", nu=nu_int,
                             mu=mu_int,
                             sigma=sigma_score)
        
        sigma = HalfNormal("$\\sigma$", sigma=score_std)
        

        # beta_red = Normal("red", 0, sigma=red_std)
        # beta_green = Normal("green", 0, sigma=green_std)
        # beta_blue = Normal("blue", mu=0, sigma=blue_std)
        # beta_asian = pymc.StudentT("Asian",nu=nu_asian, mu=0, sigma=asian_std)
        # beta_black = pymc.StudentT("Black",nu=nu_black, mu=0, sigma=black_std)
        # beta_white = pymc.StudentT("White",nu=nu_white, mu=0, sigma=white_std)
        # beta_long = pymc.StudentT("long", nu=nu_long,mu=0, sigma=long_std)
        # beta_short = pymc.StudentT("short", nu=nu_short,mu=0, sigma=short_std)
        # beta_man = pymc.StudentT("man", nu=nu_man,mu=0, sigma=man_std)
        # beta_woman = pymc.StudentT("woman",nu=nu_woman,mu=0, sigma=woman_std)
        # xr = df.red_mean.values
        # xg = df.green_mean.values
        # xb = df.blue_mean.values
        # x_asian = dummies_df['stim.race_Asian'].values
        # x_black = dummies_df['stim.race_Black'].values
        # x_white = dummies_df['stim.race_White'].values
        # x_man = dummies_df['stim.gen_men'].values
        # x_woman = dummies_df['stim.gen_women'].values

        # x_long = dummies_df.hairlength_long.values
        # x_short = dummies_df.hairlength_short.values
        
        
        # Define likelihood
        mu = intercept + \
            beta_asio * df["AsiocentricMean"].values + \
            beta_afro * df["AfrocentricMean"].values + \
            beta_euro * df["EurocentricMean"].values + \
            beta_skin * df["SkintoneMean"].values + \
            beta_masc * df["MasculinityMean"].values + \
            beta_fem * df["FemininityMean"].values + \
            beta_hair * df["HairMean"].values + \
            0
        
        
        likelihood = Normal("y", mu=mu,
                            #beta_red * xr +  beta_green * xg + beta_blue * xb +
                            #beta_black * x_black +
                            #beta_asian * x_asian +
                            #beta_white * x_white +
                            #beta_long * x_long + beta_short * x_short +
                            #beta_man * x_man + beta_woman * x_woman,
                            sigma=sigma,
                            observed=df.discriminator_score.values)
        
        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        trained_reg = sample(50000,
                             cores=8,
                             chains=10,
                             return_inferencedata=True)
        #var_names=["Black", "White","Asian", "long", "short", "man", "woman", "$\\beta_0$"]
        #var_names=["red", "green", "blue", "Black", "White","Asian", "long", "short"]        #var_names=["Black", "White","Asian", "long", "short"]
        #var_names=["red", "green", "blue"]
        var_names = categories.copy()
        var_names += ["$\\beta_0$"]
        #var_names.remove("Hair")
        #var_names.remove("Masculinity")
        #var_names.remove("Femininity")
        #var_names.remove("Skintone")
        #var_names += ["red", "green", "blue"]
        
        az.plot_trace(trained_reg,
                      var_names=var_names,
                      figsize=(7,7),
                      kind="rank_bars",
                      #compact=True,
                      #combined=True,
                      rug=False);
        plt.savefig('results/figures/ordinal_trace.svg')
        plt.savefig('results/figures/ordinal_trace.jpg')
        post_plot = az.plot_posterior(trained_reg,
                                      var_names=var_names,
                                      hdi_prob=0.95,
                                      figsize=(7,7),
                                      textsize=10,
                                      grid=(4,2))
                                      
        # ax, = pymc.plot_posterior(trained_reg,
        #                           hdi_prob=0.95,
        #                           figsize=(7,7),
        #                           kind="rank_bars")
        plt.savefig('results/figures/ordinal_posteriors.svg')
        plt.savefig('results/figures/ordinal_posteriors.jpg')
        plt.show()

