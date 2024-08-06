import unittest
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]

    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        # Absolute path example
        file_path = os.path.abspath('data/data.csv')
        self.data = pd.read_csv(filepath_or_buffer=file_path)


    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)


    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)


    def test_model_fit(
        self
    ):

        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)    

        self.model.fit(
            features=x_train,
            target=y_train
        )

        predicted_target = self.model.predict(
            x_test
        )

        report = classification_report(y_test, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30


    def test_model_predict(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        model_path = os.path.abspath("data/delay_model.pkl")

        with open(model_path, 'rb') as file:
                model = pickle.load(file)

        self.model._model = model

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)