from abc import ABC, abstractmethod
import pandas as pd 

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nData Types and Non-null Counts:")
        print(df.info())

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))

class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self.strategy = strategy

    def inspect_data(self, df: pd.DataFrame):
        self.strategy.inspect(df)
    
    def set_strategy(self, strategy: DataInspectionStrategy):
        self.strategy = strategy

if __name__ == "__main__":

    # Example
    # data_inspector = DataInspector(DataTypesInspectionStrategy())
    
    # df = pd.read_csv('C:/Users/LAKSHYA PALIWAL/Projects/extracted_data/CarPrice_Assignment.csv')
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.inspect_data(df)

    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.inspect_data(df)
    pass