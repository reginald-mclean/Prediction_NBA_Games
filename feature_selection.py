import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

def load_data(filename, interests):
    df = pd.read_csv(filename)
    df_mod = df[interests]
    data = df_mod.values
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def feature_select(features, labels):
    fs = SelectKBest(score_func=f_classif, k=15)
    x = fs.fit_transform(features, labels)
    print(x.shape)
    return x


schedule = pd.read_csv('schedule.csv')
average_5_games = pd.read_csv('2012-18_teamBoxScore_15_game_average.csv')

average_5_games_combined = []
dates = average_5_games['gmDate'].values
delete_indices = [0, 1, 2, 57, 58, 59, 60]
delete_indices.reverse()
remove = ['gmTime', 'seasTyp', 'offLNm1', 'offFNm1','offLNm2', 'offFNm2', 'offLNm3', 'offFNm3', 'teamConf', 'teamDiv', 'teamLoc','teamMin']
for r in remove:
    del average_5_games[r]
for row in schedule.itertuples():
    if row.GAME_DATE_EST in dates:
        l1 = average_5_games[(average_5_games['gmDate'] == row.GAME_DATE_EST) & (average_5_games['teamAbbr'] ==
                                                                                 row.HOME_TEAM_ID)].values.tolist()[0]
        l1.extend(average_5_games[(average_5_games['gmDate'] == row.GAME_DATE_EST) & (average_5_games['teamAbbr'] ==
                                                                            row.VISITOR_TEAM_ID)].values.tolist()[0])
        for i in delete_indices:
            del l1[i]
        if l1[0] == 'Win':
            l1.append(1)
        else:
            l1.append(0)
        del l1[0]
        average_5_games_combined.append(l1)


cols = ['HteamDayOff', 'HteamPTS',
       'HteamAST', 'HteamTO', 'HteamSTL', 'HteamBLK', 'HteamPF', 'HteamFGA',
       'HteamFGM', 'HteamFG', 'Hteam2PA', 'Hteam2PM', 'Hteam2P', 'Hteam3PA',
       'Hteam3PM', 'Hteam3P', 'HteamFTA', 'HteamFTM', 'HteamFT', 'HteamORB',
       'HteamDRB', 'HteamTRB', 'HteamPTS1', 'HteamPTS2', 'HteamPTS3', 'HteamPTS4',
       'HteamPTS5', 'HteamPTS6', 'HteamPTS7', 'HteamPTS8', 'HteamTREB', 'HteamASST',
       'HteamTS', 'HteamEFG', 'HteamOREB', 'HteamDREB', 'HteamTO_1', 'HteamSTL_1',
       'HteamBLK_1', 'HteamBLKR', 'HteamPPS', 'HteamFIC', 'teamFIC40', 'HteamOrtg',
       'HteamDrtg', 'HteamEDiff', 'HteamPlay', 'HteamAR', 'HteamASTTO', 'HteamSTLTO',
       'Hposs', 'Hpace', 'Hwin%', 'AteamDayOff', 'AteamPTS',
       'AteamAST', 'AteamTO', 'AteamSTL', 'AteamBLK', 'AteamPF', 'AteamFGA',
       'AteamFGM', 'AteamFG', 'Ateam2PA', 'Ateam2PM', 'Ateam2P', 'Ateam3PA',
       'Ateam3PM', 'Ateam3P', 'AteamFTA', 'AteamFTM', 'AteamFT', 'AteamORB',
       'AteamDRB', 'AteamTRB', 'AteamPTS1', 'AteamPTS2', 'AteamPTS3', 'AteamPTS4',
       'AteamPTS5', 'AteamPTS6', 'AteamPTS7', 'AteamPTS8', 'AteamTREB', 'AteamASST',
       'AteamTS', 'AteamEFG', 'AteamOREB', 'AteamDREB', 'AteamTO_1', 'AteamSTL_1',
       'AteamBLK_1', 'AteamBLKR', 'AteamPPS', 'AteamFIC', 'teamFIC40', 'AteamOrtg',
       'AteamDrtg', 'AteamEDiff', 'AteamPlay', 'AteamAR', 'AteamASTTO', 'AteamSTLTO',
       'Aposs', 'Apace', 'Awin%', 'target']
new_df = pd.DataFrame(data=average_5_games_combined, columns=cols)
feat_5 = new_df[['HteamDayOff', 'HteamPTS', 'HteamAST', 'HteamTO', 'HteamSTL', 'HteamBLK', 'HteamPF', 'HteamFGA',
                 'HteamFGM', 'HteamFG', 'Hteam2PA', 'Hteam2PM', 'Hteam2P', 'Hteam3PA',
       'Hteam3PM', 'Hteam3P', 'HteamFTA', 'HteamFTM', 'HteamFT', 'HteamORB',
       'HteamDRB', 'HteamTRB', 'HteamPTS1', 'HteamPTS2', 'HteamPTS3', 'HteamPTS4',
       'HteamPTS5', 'HteamPTS6', 'HteamPTS7', 'HteamPTS8', 'HteamTREB', 'HteamASST',
       'HteamTS', 'HteamEFG', 'HteamOREB', 'HteamDREB', 'HteamTO_1', 'HteamSTL_1',
       'HteamBLK_1', 'HteamBLKR', 'HteamPPS', 'HteamFIC', 'teamFIC40', 'HteamOrtg',
       'HteamDrtg', 'HteamEDiff', 'HteamPlay', 'HteamAR', 'HteamASTTO', 'HteamSTLTO',
       'Hposs', 'Hpace', 'Hwin%', 'AteamDayOff', 'AteamPTS',
       'AteamAST', 'AteamTO', 'AteamSTL', 'AteamBLK', 'AteamPF', 'AteamFGA',
       'AteamFGM', 'AteamFG', 'Ateam2PA', 'Ateam2PM', 'Ateam2P', 'Ateam3PA',
       'Ateam3PM', 'Ateam3P', 'AteamFTA', 'AteamFTM', 'AteamFT', 'AteamORB',
       'AteamDRB', 'AteamTRB', 'AteamPTS1', 'AteamPTS2', 'AteamPTS3', 'AteamPTS4',
       'AteamPTS5', 'AteamPTS6', 'AteamPTS7', 'AteamPTS8', 'AteamTREB', 'AteamASST',
       'AteamTS', 'AteamEFG', 'AteamOREB', 'AteamDREB', 'AteamTO_1', 'AteamSTL_1',
       'AteamBLK_1', 'AteamBLKR', 'AteamPPS', 'AteamFIC', 'teamFIC40', 'AteamOrtg',
       'AteamDrtg', 'AteamEDiff', 'AteamPlay', 'AteamAR', 'AteamASTTO', 'AteamSTLTO',
       'Aposs', 'Apace', 'Awin%']]
lab_5 = new_df[['target']]
selector = SelectKBest(f_classif, k=15)
selector.fit(new_df.iloc[:, :106], new_df.iloc[:, 106:])
transformed_data = selector.transform(new_df.iloc[:, :106])
mask = selector.get_support()
new_features = []
for bool, feature in zip(mask, cols):
    if bool:
        new_features.append(feature)

# unscaled data
final_df = pd.DataFrame(transformed_data, columns=new_features)
final_df['target'] = new_df['target']
final_df.to_csv('unnormalized_15_game_average_final.csv')
scaler = Normalizer()
transformed_data = scaler.fit_transform(transformed_data)
final_df = pd.DataFrame(transformed_data, columns=new_features)
final_df['target'] = new_df['target']
final_df.to_csv('normalized_15_game_average_final.csv')
print(final_df.head(20))
print(average_5_games_combined[0][31])
print(average_5_games_combined[0][32])
print(average_5_games_combined[0][33])
print(average_5_games_combined[0][34])

