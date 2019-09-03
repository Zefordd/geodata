import pickle

import pandas as pd

from utils import *


def preprocess_data(users: pd.DataFrame, geodata: pd.DataFrame, min_size: int = 50):
    ''' Предобработка полученных датафреймов.
    
    Функция находит данные, нужные для кластеризации и классификации, 
    удаляет пользователей с недостаточным количеством геометок и дней.

    Parameters
    ----------
    users: датафрейм с данными о пользователе.

    geodata: датафрейм с данными о геопозиции пользователей.

    min_size: минимальное количество геометок, собранных по пользователю,
    необходимых для кластеризации

    Returns
    -------
    users: предобработынный датафрейм с данными о пользователе.

    geodata: предобработынный датафрейм с данными о геопозиции пользователей.

    '''
    users['reg_date_time'] = pd.to_datetime(users['reg_date_time'], format='%Y-%m-%d %H:%M:%S')
    users['first_date_time'] = pd.to_datetime(users['first_date_time'], format='%Y-%m-%d %H:%M:%S')
    geodata['date_time'] = pd.to_datetime(geodata['date_time'], format='%Y-%m-%d %H:%M:%S')

    unique_users = set(users['user_id'].unique())
    unique_users.intersection_update(set(geodata['user_id'].unique()))
    list_of_users = []
    number_of_days = 3

    for user in unique_users:
        user_geodata = geodata[geodata['user_id'] == user].sort_values(by=['date_time'])
        time_delta = int(users[users['user_id'] == user]['time_delta'])
        user_geodata['date_time'] += pd.to_timedelta(str(time_delta) +  'minutes')

        user_geodata['day'] = user_geodata.date_time.dt.day
        user_geodata['hour'] = user_geodata['date_time'].dt.hour.values

        first_user_geodata = user_geodata['date_time'].iloc[0]
        # дропаем геометки пользователя, которые старше трех дней
        user_geodata.drop(user_geodata[user_geodata['date_time'] >
                                       first_user_geodata + pd.to_timedelta(str(number_of_days - 1) + 'days')].index,
                          inplace=True)
        # индексируем геометки пользователя
        user_geodata['coord_index'] = [i + 1 for i in range(user_geodata.shape[0])]

        if user_geodata.shape[0] >= min_size and len(user_geodata['day'].unique()) == 3:
            list_of_users.append(user_geodata)

    geodata = pd.DataFrame()
    geodata = pd.concat(list_of_users)

    unique_users = list(geodata['user_id'].unique())

    users = users[users['user_id'].isin(unique_users)]

    return (users, geodata)


def clustering_geodata(geodata: pd.DataFrame, max_distance: float = 0.25, min_samples: int = 50):
    ''' Кластеризация геоданных по каждому пользователю

    Parameters
    ----------
    geodata: датафрейм с данными о геопозиции пользователей.

    max_distance: максимальное расстояние между двумя точками
    в кластере

    Returns
    -------
    geodata c добавленным индексом

    '''
    list_of_users = []
    for user in geodata['user_id'].unique():
        user_data = geodata[geodata['user_id'] == user]
        user_data.insert(loc=len(user_data.columns),
                         column='cluster_id',
                         value=get_hot_spots(max_distance, min_samples, user_data).labels_)
        list_of_users.append(user_data)

    return pd.concat(list_of_users)


def calculate_factors(users: pd.DataFrame, geodata: pd.DataFrame):
    rows_list = []
    unique_user_ids = geodata['user_id'].unique()

    for user_id in unique_user_ids:
        unique_clusters_ids = geodata[geodata['user_id'] == user_id].cluster_id.unique()

        age = users.loc[users['user_id'] == user_id, 'age'].values[0]
        locale = users.loc[users['user_id'] == user_id, 'locale'].values[0]
        device = users.loc[users['user_id'] == user_id, 'device'].values[0]

        for cluster_id in unique_clusters_ids:
            if cluster_id == -1:
                continue

            factors = {}
            cluster_df = geodata[(geodata['user_id'] == user_id) & (geodata['cluster_id'] == cluster_id)]
            cluster_lat, cluster_lon = get_centroid_of_cluster(cluster_df)

            factors['cluster_lat'] = cluster_lat
            factors['cluster_lon'] = cluster_lon

            factors['user_id'] = user_id
            factors['cluster_id'] = cluster_id

            factors['max_time_in_zone'] = calculate_max_time_in_zone(cluster_df)

            unique_days = cluster_df['day'].unique()
            day_count = 1

            for day in unique_days:
                factors[f'time_day{day_count}'] = calculate_max_time_in_zone(cluster_df[cluster_df['day'] == day])
                factors[f'request_freq_{day_count}'] = find_median(cluster_df[cluster_df['day'] == day])
                day_count += 1

            while day_count < 4:
                factors[f'time_day{day_count}'] = 0
                factors[f'request_freq_{day_count}'] = 0
                day_count += 1
            
            factors['request_freq'] = find_median(cluster_df)

            factors['age'] = age

            factors['locale'] = locale

            for hour in range(0, 24):
                cluster_df_hour = geodata[
                    (geodata['user_id'] == user_id) & (geodata['cluster_id'] == cluster_id) & (geodata['hour'] == hour)]
                num_of_points = count_points(cluster_df_hour)

                factors[f'is_points_in_{hour}'] = num_of_points if num_of_points > 0 else 0
                factors[f'points_in_{hour}'] = count_points(cluster_df_hour)

            factors['top_reason'] = cluster_df['reason'].mode().values[0]

            factors['top_source'] = cluster_df['source'].mode().values[0]

            cluster_df_charge = geodata[
                (geodata['user_id'] == user_id) & (geodata['cluster_id'] == cluster_id) & (geodata['is_charge'] == 1)]
            factors['charge_time'] = calculate_max_time_in_zone(cluster_df_charge)

            cluster_df_wifi = geodata[
                (geodata['user_id'] == user_id) & (geodata['cluster_id'] == cluster_id) & (geodata['source'] == 'wifi')]
            factors['wifi_time'] = calculate_max_time_in_zone(cluster_df_wifi)

            factors['charge_connections'] = count_connection_seq(cluster_df_charge)

            factors['wifi_connections'] = count_connection_seq(cluster_df_wifi)

            factors['hour_mode'] = cluster_df['hour'].mode()[0]

            factors['device'] = device

            factors['max_distance'] = find_max_distance(cluster_df, cluster_lat, cluster_lon)

            rows_list.append(factors)

    return pd.DataFrame(rows_list)


def predict_category_id(factors: pd.DataFrame, clf):
    factors = preprocess_factors(factors)
    columns_to_normalize = create_columns_to_normalize()

    clusters_info = factors[['user_id', 'cluster_id', 'cluster_lat', 'cluster_lon', 'max_distance']]

    for col in columns_to_normalize.keys():
        factors[col] = (factors[col] - factors[col].mean()) / factors[col].std()

    factors.fillna(0, inplace=True)

    factors = factors.drop(['user_id', 'cluster_id', 'cluster_lat', 'cluster_lon'], axis=1)
    factors = factors.drop(['charge_connections', 'charge_time'], axis=1)  # delete me
    X = factors.values

    result = clf.predict(X)
    places = {}
    unique_users = clusters_info.user_id.unique()
    j = 0
    for user_id in unique_users:
        i = 1
        user_clusters = clusters_info[clusters_info['user_id'] == user_id]
        places[f'user_{user_id}'] = {}
        for _, cluster in user_clusters.iterrows():
            radius = cluster['max_distance'] if cluster['max_distance'] >= 50 else 50.0 
            places[f'user_{user_id}'][f'cluster{i}'] = {
                                                            'lat': cluster['cluster_lat'],
                                                            'lon': cluster['cluster_lon'],
                                                            'radius': radius,
                                                            'category': result[j],
                                                        }
            i += 1
            j += 1

    return places


def find_places(users: pd.DataFrame, geodata: pd.DataFrame, clf_path: str = 'model/clf.pickle'):
    users, geodata = preprocess_data(users, geodata)
    geodata = clustering_geodata(geodata)
    factors = calculate_factors(users, geodata)

    with open(clf_path, 'rb') as handle:
        clf = pickle.load(handle)

    places = predict_category_id(factors, clf)

    return places
