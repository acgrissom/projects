import csv
import statistics
from collections import defaultdict
#with open("data/cleaned/Study2_GoogleFaces_full_unbalanced.csv") as csvfile:
with open("data/cleaned/Study2_GoogleFaces_cleaned_50pergroup.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    out = open('data/cleaned/Study2_50_aggregated.csv', 'w')
    categories = ["Masculinity",
                  "Femininity",
                  "Hair",
                  "Skintone",
                  "Afrocentric",
                  "Eurocentric",
                  "Asiocentric"]
    fieldnames = reader.fieldnames
    for cat in categories:
        fieldnames += [cat + "Std", cat + "Mean"]
    csvwriter = csv.DictWriter(out, fieldnames)
    csvwriter.writeheader()
    for row in reader:
        skip = False
        row_dup = row.copy()
        values_dict = defaultdict(list)
        for col in row.keys():
            for cat in categories:
                if col.startswith(cat):
                    #print(col + " " + cat)
                    if row[col] != "NA" and row[col] is not None:
                        values_dict[cat] += [float(row[col])]
        #print(values_dict)
        for cat in categories:
            cat_vals = values_dict[cat]
            if len(cat_vals) == 3:
                row_dup[cat + "Std"] = statistics.stdev(cat_vals)
                row_dup[cat + "Mean"] = statistics.mean(cat_vals)
            else:
                skip = True

        if not skip:
            csvwriter.writerow(row_dup)
            
