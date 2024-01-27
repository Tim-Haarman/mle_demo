from pathlib import Path



class IOConfig:
    _root_folder = Path(__file__).parents[2]
    data_path = f'{_root_folder}/data/bank_data_train.csv'
    encoder_path = f'{_root_folder}/artifacts/encoders/encoder.pkl'
    model_folder_path = f'{_root_folder}/artifacts/models/'


class PreprocessingConfig:
    feature_columns = [
        "CreditScore", 
        "Geography", 
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary"	
    ]
    categorical_features = [
        "Geography", 
        "Gender",
    ]
    target_column = "Exited"
    


class ModelConfig:
    test_size = 0.2
    model_kwargs = {
        "n_estimators": 100
    }