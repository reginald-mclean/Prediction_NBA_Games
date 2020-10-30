import numpy as np
import pandas as pd

def load_data(game_avg_counter=5):
    data = np.genfromtxt("./Data/archive_2012_2018/2012-18_teamBoxScore.csv", delimiter=',', dtype=None,
                         encoding=None, names=True)
    df = pd.DataFrame(data)
    print(df.head())
    df['gmDate'] = pd.to_datetime(df.gmDate)
    df = df.sort_values(by='gmDate')
    df['win%'] = np.nan
    team_ids_dict = {}
    team_ids_list = []
    data_columns = list(df.columns)
    print(data_columns)
    print(len(data_columns))
    key_stat_1 = data_columns[14:64+1]
    key_stat_2 = data_columns[70:]
    key_stat_col = key_stat_1 + key_stat_2  # all the columns that need to be averaged
    print(key_stat_1)
    print(key_stat_2)
    print(key_stat_col)
    for index, row in df.iterrows():
        game_date = pd.to_datetime(row['gmDate'])
        if row['teamRslt'] == 'Win':
            row['win%'] = 1
        else:
            row['win%'] = 0
        df.loc[index] = row
        print(game_date)
        a = row['teamAbbr']
        if a in team_ids_list:
            team_ids_dict[a][game_date] = row[key_stat_col].to_numpy().copy()
        else:
            team_ids_list.append(a)
            team_ids_dict[a] = {}
            team_ids_dict[a][game_date] = row[key_stat_col].to_numpy().copy()
    # print(team_ids_list)
    # print(team_ids_dict['TOR']['2012-10-31'])
    # print(len(team_ids_dict['TOR']['2012-10-31']))
    # print(len(key_stat_col))

    for index, row in df.iterrows():
        game_date = row['gmDate']
        team_id = row['teamAbbr']
        result = calculate_x_games_back(team_ids_dict[team_id], game_avg_counter, game_date)
        row[key_stat_col] = result
        df.loc[index] = row
    df = df.dropna()
    print(df.head())
    return df


def calculate_x_games_back(team_dict, x_games, game_date):
    results = np.zeros(shape=(105,), dtype=np.float32)
    team_game_dates = list(team_dict.keys())
    dates = []
    for date in team_game_dates:
        if date < game_date:
            dates.append(date)
    dates = dates[-x_games:]
    for date in dates:
        results = np.add(results, team_dict[date])
    print(dates)
    return np.divide(results, len(dates))


if __name__ == '__main__':
    games_df = load_data()
    games_df.to_csv('./Data/archive_2012_2018/averaged_data/2012-18_teamBoxScore_5_game_average.csv')
    games_df = load_data(game_avg_counter=10)
    games_df.to_csv('./Data/archive_2012_2018/averaged_data/2012-18_teamBoxScore_10_game_average.csv')
    games_df = load_data(game_avg_counter=15)
    games_df.to_csv('./Data/archive_2012_2018/averaged_data/2012-18_teamBoxScore_15_game_average.csv')
