from abc import ABC, abstractmethod
import pandas as pd
import zipfile
import os

class DataIngestor(ABC):
    @abstractmethod
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestor):
    def ingest_data(self, file_path: str) -> pd.DataFrame:

        if not file_path.endswith(".zip"):
            raise ValueError("Given file is not a .zip file...")
        
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")
        
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]


        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data")
        if len(csv_files) > 1:
            raise ValueError("Multiple csv files found please select one....")
        
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)
        print(f"Data ingested from {file_path}")

        return df
    
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No DataIngestor found for the given file format: {file_extension}")
        

if __name__ == "__main__":
    file_path = 'C:/Users/LAKSHYA PALIWAL/Projects/car-price-prediction/data/archive (1).zip'

    file_extension = os.path.splitext(file_path)[1]

    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    df = data_ingestor.ingest_data(file_path)

    print(df.head())