import pandas as pd


class NYLoader:
    def __init__(self, airbnb_path, taxi_path=None, link=False):
        print("Loading airbnb from {}".format(airbnb_path))
        self.airbnb_data = pd.read_csv(airbnb_path)
        print("Loaded.")
        if taxi_path is not None:
            print("Loading taxi from {}".format(taxi_path))
            self.taxi_data = pd.read_csv(taxi_path)
            print("Loaded.")

        if link:
            self.labels = self.airbnb_data['price'].to_numpy()
            self.airbnb_data.drop(columns=['price'], inplace=True)

            # move lon and lat to end of airbnb
            ab_cols = list(self.airbnb_data)
            ab_cols.insert(len(ab_cols), ab_cols.pop(ab_cols.index('longitude')))
            ab_cols.insert(len(ab_cols), ab_cols.pop(ab_cols.index('latitude')))
            self.airbnb_data = self.airbnb_data[ab_cols]
            print("Current aribnb columns: " + str(list(self.airbnb_data)))
            self.airbnb_data = self.airbnb_data.to_numpy()

            # move lon and lat to the front of taxi
            tx_cols = list(self.taxi_data)
            tx_cols.insert(0, tx_cols.pop(tx_cols.index('lat')))
            tx_cols.insert(0, tx_cols.pop(tx_cols.index('lon')))
            self.taxi_data = self.taxi_data[tx_cols]
            print("Current taxi columns: " + str(list(self.taxi_data)))
            self.taxi_data = self.taxi_data.to_numpy()
        else:
            self.airbnb_data.drop(columns=['longitude', 'latitude'], inplace=True)
            self.labels = self.airbnb_data['price'].to_numpy()
            self.airbnb_data = self.airbnb_data.drop(columns=['price']).to_numpy()

    def load_single(self):
        return self.airbnb_data, self.labels

    def load_parties(self):
        return [self.airbnb_data, self.taxi_data], self.labels
