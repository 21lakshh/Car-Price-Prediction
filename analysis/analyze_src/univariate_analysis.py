from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class UnivariteAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass

class NumericalUnivariateAnalysis(UnivariteAnalysisStrategy):
    def analyze(self, df, feature: str):
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=True, bins=50)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

class CategoricalUnivariateAnalysis(UnivariteAnalysisStrategy):
        def analyze(self, df, feature: str):
            plt.figure(figsize=(6,4))
            sns.countplot(x=feature,data=df,palette="muted")
            plt.title(f"Distruibution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()

class UnivariateAnalysis:
    def __init__(self, strategy: UnivariteAnalysisStrategy):
        self.strategy = strategy

    def analyze(self, df, feature: str):
        self.strategy.analyze(df, feature)
    
    def set_strategy(self, strategy: UnivariteAnalysisStrategy):
        self.strategy = strategy

if __name__ == "__main__":
    # Testing the Univariate Analysis
    # df = pd.read_csv("C:/Users/LAKSHYA PALIWAL/Projects/extracted_data/CarPrice_Assignment.csv")

    # univariate_analysis = UnivariateAnalysis(NumericalUnivariateAnalysis())
    # univariate_analysis.analyze(df, "price")

    # univariate_analysis.set_strategy(CategoricalUnivariateAnalysis())
    # univariate_analysis.analyze(df, "fueltype")
    pass