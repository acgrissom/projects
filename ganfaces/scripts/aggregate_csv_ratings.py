import csv
import pandas
from collections import defaultdict
import pandasql
from pandasql import sqldf
FILENAME = "data/Study2_GoogleFaces_cleaned.csv"
df = pandas.read_csv(FILENAME)
df = df.sample(frac=1) # shuffle rows
race_cols = [df.columns.get_loc(col) for col in df.columns if ('Race.' or 'Race ') in col]
RACES = ["Asian", "Black", "White"]

def count_races(df_row) -> dict:
    race_counts = defaultdict(int)
    for race_col in race_cols:
        race = df_row[race_col]
        if not type(race) is float:
            race_counts[race.split(" ")[0]] += 1
    return race_counts

def race_percentages(race_counts : dict) -> dict:
    for race in RACES:
        if not race in race_counts:
            race_counts[race] = 0
    
    sorted_races = sorted(race_counts.items(), key=lambda kv: kv[1], reverse=True)
    total_counts = sum(race_counts.values())
    race_percents = dict(race_counts)
    for i in range(len(sorted_races)):
        for race in race_percents.keys():
            if sorted_races[i][0].startswith(race):
                race_percents[race] /= total_counts
    return race_percents
    
def top_race_and_percentage(race_counts: dict) -> tuple:
    sorted_races = sorted(race_counts.items(), key=lambda kv: kv[1], reverse=True)
    total_counts = sum(race_counts.values())   
    percent = sorted_races[0][1] / total_counts
    return sorted_races[0][0], percent
    

if __name__ == "__main__":
    new_df = pandas.DataFrame()
    new_col_names = ["percent_Asian", "percent_Black", "percent_White", "top_race", "top_race_percent"]
    new_vals = defaultdict(str)
    
    # create new columns

    for name in new_col_names:
        df[name] = 0
        #percent_Asian, etc.

    new_df = pandas.DataFrame(df)

    for index, row in df.iterrows():
        race_counts = count_races(row)
        # skip rows without race info
        if not len(race_counts.keys()) == 0:
            top_race, top_race_percentage = top_race_and_percentage(race_counts)
            race_percents = race_percentages(race_counts)
            df.at[index, "top_race"] = top_race
            df.at[index, "top_race_percent"] = top_race_percentage
            df.at[index, "percent_Black"] = race_percents["Black"]
            df.at[index, "percent_White"] = race_percents["White"]
            df.at[index, "percent_Asian"] = race_percents["Asian"]
    #df.to_csv(')
         # get non top, bottom images
    query = """
    SELECT *
    FROM df
    WHERE image NOT LIKE 'top%' AND
    image NOT LIKE 'bot%'
    """
            
    print(df.head())
