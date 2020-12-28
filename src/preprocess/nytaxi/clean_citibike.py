import os
import sys

import pandas as pd


def clean_bike(bike_ori_data_path, out_bike_data_path, sample_n=None, keep_col=None):
    print("Reading from {}".format(bike_ori_data_path))
    bike_ori_data = pd.read_csv(bike_ori_data_path, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    print("get pick-up and drop-off hour")
    bike_ori_data.drop(columns=['store_and_fwd_flag'], inplace=True)

    print("get pick-up and drop-off hour")
    bike_ori_data['pickup_hour'] = bike_ori_data['tpep_pickup_datetime'].dt.hour
    bike_ori_data['dropoff_hour'] = bike_ori_data['tpep_dropoff_datetime'].dt.hour

    print("drop specific time information")
    bike_ori_data.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

    print("divide pickup and dropoff dataset")
    bike_ori_data_pickup = bike_ori_data.drop(columns=['dropoff_hour', 'dropoff_longitude', 'dropoff_latitude'])
    bike_ori_data_pickup['is_pickup'] = 1
    bike_ori_data_pickup.rename(columns={'pickup_hour': 'hour',
                                        'pickup_longitude': 'lon',
                                        'pickup_latitude': 'lat'}, inplace=True)
    bike_ori_data_dropoff = bike_ori_data.drop(columns=['pickup_hour', 'pickup_longitude', 'pickup_latitude'])
    bike_ori_data_dropoff.rename(columns={'dropoff_hour': 'hour',
                                         'dropoff_longitude': 'lon',
                                         'dropoff_latitude': 'lat'}, inplace=True)
    bike_ori_data_dropoff['is_pickup'] = 0

    print("concat pickup and dropoff dataset by rows")
    out_bike_data = pd.concat([bike_ori_data_pickup, bike_ori_data_dropoff])
    print("Finished, print all the columns:")
    print(out_bike_data.dtypes)

    if keep_col is None:
        print("make categorical features one-hot")
        out_bike_data = pd.get_dummies(out_bike_data,
                                      columns=['hour', 'VendorID', 'RatecodeID', 'payment_type'],
                                      prefix=['hr', 'vid', 'rid', 'pt'], drop_first=True)
    else:
        print("Filter columns {}".format(keep_col))
        out_bike_data = out_bike_data[keep_col + ['lon', 'lat']]
        print("make categorical features one-hot")
        dummy_col, dummy_prefix = [], []
        col_prefix = {
            'hour': 'hr',
            'VendorID': 'vid',
            'RatecodeID': 'rid',
            'payment_type': 'pt'
        }
        for col, prefix in col_prefix.items():
            if col in out_bike_data.columns:
                dummy_col.append(col)
                dummy_prefix.append(prefix)
        out_bike_data = pd.get_dummies(out_bike_data, columns=dummy_col, prefix=dummy_prefix, drop_first=True)

    print("sampling from dataset")
    if sample_n is not None:
        out_bike_data = out_bike_data.sample(n=sample_n, random_state=0)

    print("Saving cleaned dataset to {}".format(out_bike_data_path))
    out_bike_data.to_csv(out_bike_data_path, index=False)
    print("Saved {} samples to file".format(len(out_bike_data.index)))


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/nytaxi")  # change working directory
    # clean_bike("yellow_tripdata_2016-06.csv", "taxi_201606_clean.csv", sample_n=None)
    clean_bike("yellow_tripdata_2016-06.csv", "taxi_201606_clean_sample_1e6.csv",
              sample_n=1000000, keep_col=['RatecodeID', 'tip_amount'])
