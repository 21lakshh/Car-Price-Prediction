import logging 
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LogTransformation(FeatureEngineeringStrategy):

    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:


        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])

        logging.info("Log transformation completed.")
        return df_transformed
    
class StandardScaling(FeatureEngineeringStrategy):

    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:

        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        scaler = StandardScaler()
        df_transformed[self.features] = scaler.fit_transform(df[self.features])

        logging.info("Standard scaling completed.")
        return df_transformed

class MinMaxScaling(FeatureEngineeringStrategy):

    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
            
            logging.info(f"Applying min-max scaling to features: {self.features}")
            df_transformed = df.copy()
            scaler = MinMaxScaler()
            df_transformed[self.features] = scaler.fit_transform(df[self.features])
    
            logging.info("Min-max scaling completed.")
            return df_transformed
    
# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoder = OneHotEncoder()
        df_transformed[self.features] = encoder.fit_transform(df[self.features])
        
        logging.info("One-hot encoding completed.")
        return df_transformed
    

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying feature engineering strategy.")
        return self.strategy.apply_transformation(df)
    
# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    pass
    
    