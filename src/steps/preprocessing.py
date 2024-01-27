import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from config import PreprocessingConfig, IOConfig


# Data preprocessing
class Preprocessor:
    def __init__(self, data: pd.DataFrame, inference_mode: bool):
        self.data = data
        self.labels = self.data.pop(PreprocessingConfig.target_column)
        self.inference_mode = inference_mode

    def __call__(self):
        self.select_features()
        encoder = self.get_encoder()
        self.encode_data(encoder=encoder)
        return self.data, self.labels

    def select_features(self):
        self.data = self.data[PreprocessingConfig.feature_columns]

    def get_encoder(self):
        if self.inference_mode:
            return self._load_encoder()
        return self._fit_encoder()

    def encode_data(self, encoder):
        self.data = encoder.transform(self.data)

    def _fit_encoder(self):
        encoder = ColumnTransformer(
            transformers=[
                (
                    "onehot",
                    OneHotEncoder(drop="if_binary"),
                    PreprocessingConfig.categorical_features,
                )
            ],
            remainder="passthrough",
        )
        encoder.fit(self.data)
        self._save_encoder(encoder=encoder)
        return encoder

    def _load_encoder(self):
        with open(IOConfig.encoder_path, "rb") as encoder_file:
            return pickle.load(encoder_file)

    def _save_encoder(self, encoder):
        with open(IOConfig.encoder_path, "wb") as encoder_file:
            pickle.dump(encoder, encoder_file)


if __name__ == "__main__":
    df = pd.read_csv(IOConfig.data_path)
    processor = Preprocessor(data=df, inference_mode=False)

    print(processor())
