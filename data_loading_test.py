import numpy as np
import pandas as pd
import collections as c


data = np.genfromtxt("../games.csv", delimiter=',', dtype=None, encoding=None, names=True)
df = pd.DataFrame(data)

ids = df['HOME_TEAM_ID'].unique()
team_ids_dict = {}
for id in ids:
    team_dates = []
    for index, row in df.iterrows():
        if row['HOME_TEAM_ID'] == id:
            team_dates.append(row['GAME_DATE_EST'])
    team_dates.sort()
    ordered_dictionary = c.OrderedDict()
    for date in team_dates:
        ordered_dictionary[date] = df[(df['HOME_TEAM_ID'] == id) & (df['GAME_DATE_EST'] == date)]
    team_ids_dict[id] = ordered_dictionary

# need to now use OG data to build data pipeline that can take parameter X and generate data going back X days



# testing
