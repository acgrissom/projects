#https://fractalytics.io/visualization-scikit-learn-decision-trees-d3-js

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
import dtreeviz
df = pd.read_csv('data/cleaned/Study2_unbalanced_aggregated.csv')
#df = pd.read_csv('data/cleaned/Study2_50_aggregated.csv')


categories = ["Masculinity",
              "Femininity",
              "Skintone",
              "Afrocentric",
              "Eurocentric",
              "Asiocentric",
              "Hair"]



def clean_data(df,
               use_binary=False,
               use_rgb=False,
               use_ordinal=False,
               y="discriminator_score") -> tuple:
    x_columns_dict = {}
    if use_ordinal:
        for cat in categories:
            #std_str = cat + "Std"
            #x_columns_dict[std_str] = df[std_str].values
            mean_str = cat + "Mean"
            x_columns_dict[mean_str] = df[mean_str].values

        
    if use_rgb:
        x_columns_dict["red_mean"] = df.red_mean.values
        x_columns_dict["green_mean"] = df.green_mean.values
        x_columns_dict["blue_mean"] = df.blue_mean.values
        
    if use_binary:
        dummies_df = pd.get_dummies(df)
        x_columns_dict['LongHair'] = dummies_df.hairlength_long.values
        x_columns_dict['ShortHair'] = dummies_df.hairlength_short.values
        x_columns_dict['Asian'] = dummies_df['stim.race_Asian'].values
        x_columns_dict['Black'] = dummies_df['stim.race_Black'].values
        x_columns_dict['White'] = dummies_df['stim.race_White'].values
        x_columns_dict['Man'] = dummies_df['stim.gen_men'].values
        x_columns_dict['Woman'] = dummies_df['stim.gen_women'].values



    y = pd.DataFrame(df[y].values)
    new_df = pd.DataFrame(x_columns_dict)
    return new_df, y
    

X, y = clean_data(df,
                  use_binary=False,
                  use_rgb=False,
                  use_ordinal=True,
                  y="discriminator_score")



# Fit regression model
#reg_2 = DecisionTreeRegressor(max_depth=2)
reg = DecisionTreeRegressor(max_depth=4,
                            max_features=1.0)
scores = cross_val_score(reg, X, y, cv=5)
print(scores)
tree_model = reg.fit(X.values, y.values)

#tree.plot_tree(tree_model,
#               feature_names=X.columns)

viz_model = dtreeviz.model(model=tree_model,
                            X_train=X.values, 
                            y_train=y.values, 
                            feature_names=list(X.columns.values), 
                            target_name="score")
#viz_model.rtree_feature_space(features=['red_mean'])
print(X.columns)
viz_model.rtree_feature_space(features=['AfrocentricMean', 'MasculinityMean'])                          
#viz_model.view(fancy=False)

plt.savefig("results/figures/tree.svg")
plt.savefig("results/figures/tree.png")
#plt.show()

exit()
# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
