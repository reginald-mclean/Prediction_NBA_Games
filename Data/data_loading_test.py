import numpy as np
import pandas as pd


def load_data():
    data = np.genfromtxt("../games.csv", delimiter=',', dtype=None, encoding=None, names=True)
    df = pd.DataFrame(data)
    df['GAME_DATE_EST'] = pd.to_datetime(df.GAME_DATE_EST)
    df = df.sort_values(by='GAME_DATE_EST')
    team_ids_dict = {}
    home_columns = ["PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home"]
    visitor_columns = ["PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away"]
    team_ids_list = []
    indexes = df.index.copy()
    df['index'] = indexes
    df.set_index('index')
    df = df.dropna()

    for index, row in df.iterrows():
        # print(row['GAME_DATE_EST']) # This is just to show progress, it can take awhile lol
        a = row['HOME_TEAM_ID']
        if a in team_ids_list:
            team_ids_dict[a][row['GAME_DATE_EST']] = row[home_columns].to_numpy().copy()
        else:
            team_ids_list.append(a)
            team_ids_dict[a] = {}
            team_ids_dict[a][row['GAME_DATE_EST']] = row[home_columns].to_numpy().copy()
        a = row['VISITOR_TEAM_ID']
        if a in team_ids_list:
            team_ids_dict[a][row['GAME_DATE_EST']] = row[visitor_columns].to_numpy().copy()
        else:
            team_ids_list.append(a)
            team_ids_dict[a] = {}
            team_ids_dict[a][row['GAME_DATE_EST']] = row[visitor_columns].to_numpy().copy()

    for index, row in df.iterrows():
        game_date = pd.to_datetime(row['GAME_DATE_EST'])
        home_team_id = row['HOME_TEAM_ID']
        visitor_team_id = row['VISITOR_TEAM_ID']
        home_result = calculate_x_games_back(team_ids_dict[home_team_id], 5, game_date)
        row[home_columns] = home_result
        visitor_result = calculate_x_games_back(team_ids_dict[visitor_team_id], 5, game_date)
        row[visitor_columns] = visitor_result
        df.loc[index] = row
    df = df.dropna()


def calculate_x_games_back(team_dict, x_games, game_date):
    results = np.zeros(shape=(6,), dtype=np.float32)
    team_game_dates = list(team_dict.keys())
    dates = []
    for date in team_game_dates:
        if date < game_date:
            dates.append(date)
    dates = dates[-x_games:]
    for date in dates:
        results = np.add(results, team_dict[date])
    return np.divide(results, len(dates))


load_data()
