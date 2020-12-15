import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename, interests):
    df = pd.read_csv(filename)
    df_mod = df[interests]
    data = df_mod.values
    x = data[:, 1:].copy()
    y = data[:, 0].copy()
    return x, y


def feature_select(features, labels):
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(features, labels)
    return fs

#Features of Interest
# columns = ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_away', 'FG_PCT_away',
#            'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'WIN_PCT_home', 'WIN_PCT_away', 'HOME_TEAM_WINS']

columns =['teamRslt', 'teamDayOff', 'teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA', 'teamFGM', 'teamFG', 'team2PA', 'team2PM', 'team2P', 'team3PA',
          'team3PM', 'team3P', 'teamFTA', 'teamFTM', 'teamFT', 'teamORB', 'teamDRB', 'teamTRB', 'teamPTS1', 'teamPTS2', 'teamPTS3', 'teamPTS4', 'teamPTS5', 'teamPTS6', 'teamPTS7',
          'teamPTS8', 'teamTREB', 'teamASST', 'teamTS', 'teamEFG', 'teamOREB', 'teamDREB', 'teamTO_1', 'teamSTL_1', 'teamBLK_1', 'teamBLKR', 'teamPPS', 'teamFIC', 'teamFIC40',
          'teamOrtg', 'teamDrtg', 'teamEDiff', 'teamPlay', 'teamAR', 'teamASTTO', 'teamSTLTO', 'opptDayOff', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA',
          'opptFGM', 'opptFG', 'oppt2PA', 'oppt2PM', 'oppt2P', 'oppt3PA', 'oppt3PM', 'oppt3P', 'opptFTA', 'opptFTM', 'opptFT', 'opptORB', 'opptDRB', 'opptTRB', 'opptPTS1', 'opptPTS2',
          'opptPTS3', 'opptPTS4', 'opptPTS5', 'opptPTS6', 'opptPTS7', 'opptPTS8', 'opptTREB', 'opptASST', 'opptTS', 'opptEFG', 'opptOREB', 'opptDREB', 'opptTO_1', 'opptSTL_1', 'opptBLK_1',
          'opptBLKR', 'opptPPS', 'opptFIC', 'opptFIC40', 'opptOrtg', 'opptDrtg', 'opptEDiff', 'opptPlay', 'opptAR', 'opptASTTO', 'opptSTLTO', 'poss', 'pace', 'win%']

#Loading Data
feat_5, lab_5 = load_data('2012-18_teamBoxScore_5_game_average.csv', columns)
feat_10, lab_10 = load_data('2012-18_teamBoxScore_10_game_average.csv', columns)
feat_15, lab_15 = load_data('2012-18_teamBoxScore_15_game_average.csv', columns)
for i in range(len(lab_5)):
    assert(lab_5[i] == lab_10[i] == lab_15[i])
#Apply feature selection
fs_5 = feature_select(feat_5, lab_5)
fs_10 = feature_select(feat_10, lab_10)
fs_15 = feature_select(feat_15, lab_15)

#Plotting Scores
fig, (ax1) = plt.subplots(1) #, ax2, ax3)
fig.suptitle('Feature Selection Scores')
ax1.bar([i for i in range(len(fs_5.scores_))], fs_5.scores_)
#ax2.bar([i for i in range(len(fs_10.scores_))], fs_10.scores_)
#ax3.bar([i for i in range(len(fs_15.scores_))], fs_15.scores_)
for i in range(len(columns)):
    print("Column {0} is {1}".format(i, columns[i]))

ax1.set_xticks(np.arange(len(columns)))
ax1.set_xticklabels(columns, rotation = 90, ha="right")
plt.tight_layout()
plt.show()
#ax2.set_xticklabels(columns, rotation = 45, ha="right")
#plt.show()
#ax3.set_xticklabels(columns, rotation = 45, ha="right")
#plt.show()



