from math import radians, sin, cos, acos

import numpy as np
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN


def calculate_max_time_in_zone(cluster_df):
    local_max = global_max = 0
    prev_index = -1
    prev_date_time = None

    for _, row in cluster_df.iterrows():

        if row['coord_index'] == prev_index + 1:
            prev_index = row['coord_index']

            if not prev_date_time:
                prev_date_time = row['date_time']
                continue

            local_max += (row['date_time'] - prev_date_time) / np.timedelta64(1, 'm')
            prev_date_time = row['date_time']
            if local_max > global_max:
                global_max = local_max

        else:
            if local_max > global_max:
                global_max = local_max
            prev_index = row['coord_index']
            prev_date_time = row['date_time']
            local_max = 0

        if local_max > global_max:
            global_max = local_max

    return int(global_max)


def calculate_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    try:
        result = 6371008.8 * (
            acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
        )
    except ValueError:
        result = 0

    return result


def get_centroid_of_cluster(cluster_df):
    centroid = MultiPoint(cluster_df.as_matrix(columns=['lat', 'lon']))
    return (centroid.centroid.x, centroid.centroid.y)


def find_max_distance(cluster_df, cluster_lat, cluster_lon):
    max_distance = 0
    for _, row in cluster_df.iterrows():
        distance = calculate_distance(row['lon'], row['lat'], cluster_lon, cluster_lat)
        if distance > max_distance:
            max_distance = distance
    return max_distance


def count_connection_seq(cluster_df):
    count = 0
    prev_index = -1
    is_first = True

    for _, row in cluster_df.iterrows():

        if row['coord_index'] == prev_index + 1:
            prev_index = row['coord_index']

            if is_first:
                count += 1
                continue
            else:
                is_first = False
        else:
            prev_index = row['coord_index']
            is_first = False
            count += 1

    return count


def count_points(cluster_df):
    return len(cluster_df.index)


def find_median(cluster_df):
    time_deltas = []
    prev_date_time = None
    prev_index = -1

    for _, row in cluster_df.iterrows():
        if row['coord_index'] == prev_index + 1:
            prev_index = row['coord_index']

            if not prev_date_time:
                prev_date_time = row['date_time']
                continue

            time_deltas.append((row['date_time'] - prev_date_time) / np.timedelta64(1, 'm'))
            prev_date_time = row['date_time']

        else:
            prev_index = row['coord_index']
            prev_date_time = row['date_time']

    return np.median(np.asanyarray(time_deltas))


def get_hot_spots(max_distance, min_samples, user_data):
    coords = user_data.as_matrix(columns=['lat', 'lon'])
    kms_per_radian = 6371.0088
    epsilon = max_distance / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples,
                algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    return db


def preprocess_factors(factors):
    factors["top_reason"] = factors["top_reason"].astype('category')
    factors["top_reason"] = factors["top_reason"].cat.codes

    factors["top_source"] = factors["top_source"].astype('category')
    factors["top_source"] = factors["top_source"].cat.codes

    factors["device"] = factors["device"].astype('category')
    factors["device"] = factors["device"].cat.codes

    factors["age"] = factors["age"].astype('category')
    factors["age"] = factors["age"].cat.codes

    factors["locale"] = factors["locale"].astype('category')
    factors["locale"] = factors["locale"].cat.codes

    return factors


def create_columns_to_normalize():
    columns_to_normalize = {}
    columns_to_normalize['max_time_in_zone'] = {}
    columns_to_normalize['request_freq'] = {}
    columns_to_normalize['charge_time'] = {}
    columns_to_normalize['wifi_time'] = {}
    columns_to_normalize['charge_connections'] = {}
    columns_to_normalize['wifi_connections'] = {}
    columns_to_normalize['max_distance'] = {}

    for day in range(1, 4):
        columns_to_normalize[f'time_day{day}'] = {}
        columns_to_normalize[f'request_freq_{day}'] = {}

    for hour in range(0, 24):
        columns_to_normalize[f'points_in_{hour}'] = {}

    return columns_to_normalize
