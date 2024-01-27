import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from config import IOConfig, ModelConfig


#TODO make this dynamic, allow multiple model implementations. Probably some base class.
class Model:
    def __init__(self, inference_mode: bool) -> None:
        self.model_name = "randomforest"
        if inference_mode:
            self.model = self._load_model()
        else:
            self.model = None

    def predict(self, data):
        if self.model is None:
            raise ValueError("First train or load a model in non-inference mode.")
        return self.model.predict(data)

    def train_model(self, data, labels):
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=ModelConfig.test_size
        )

        self.model = RandomForestClassifier(**ModelConfig.model_kwargs)
        self.model.fit(x_train, y=y_train)

        y_pred = self.model.predict(x_test)

        print(classification_report(y_test, y_pred))


    # These two are the same as in preprocessing.py, probably extract that.
    def _load_model(self):
        with open(
            f"{IOConfig.model_folder_path}{self.model_name}.pkl", "r"
        ) as model_file:
            return pickle.load(model_file)
        
    def _save_model(self, model):
        with open(
            f"{IOConfig.model_folder_path}{self.model_name}.pkl", "wb"
        ) as model_file:
            pickle.dump(model, model_file)



if __name__ == "__main__":
    from preprocessing import Preprocessor
    import pandas as pd

    df = pd.read_csv(IOConfig.data_path)
    processor = Preprocessor(data=df, inference_mode=False)
    processed_data, labels = processor()

    model = Model(inference_mode=False)
    model.train_model(processed_data, labels=labels)
