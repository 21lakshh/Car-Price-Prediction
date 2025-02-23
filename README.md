# Car Price Prediction - End-to-End Data Science Project

## Project Overview

This project is an end-to-end machine learning pipeline for predicting car prices. It involves various stages, including data preprocessing, feature engineering, model training, and evaluation. The entire pipeline is automated using ZenML, ensuring reproducibility and scalability.

## Key Features

- **Data Ingestion:** Loading and preprocessing raw housing data.
- **Handling Missing Values:** Imputing missing data using appropriate techniques.
- **Feature Engineering:** Creating meaningful features to improve model performance.
- **Outlier Detection:** Identifying and handling outliers to avoid model biases.
- **Data Splitting:** Splitting data into training and testing sets.
- **Model Building:** Training multiple machine learning models.
- **Model Evaluation:** Comparing models using performance metrics.
- **Pipeline Automation:** Using ZenML to orchestrate and manage the workflow.

---

## ZenML and Pipeline Automation

### What is ZenML?

[ZenML](https://zenml.io/) is an extensible, open-source MLOps framework that simplifies the process of building, deploying, and managing machine learning pipelines. It ensures consistency across workflows and enables tracking and reproducibility of ML experiments.

### ZenML Pipeline Structure in This Project

This project follows a modular pipeline architecture with the following steps:

#### **Data Ingestion Step:**
- Reads raw housing data.
- Stores it in a structured format for further processing.

#### **Data Cleaning & Preprocessing Step:**
- Handles missing values.
- Detects and manages outliers.
- Encodes categorical variables and scales numerical features.

#### **Feature Engineering Step:**
- Creates new meaningful features.
- Performs transformations like log scaling and polynomial features.

#### **Data Splitting Step:**
- Splits the dataset into training and testing sets.

#### **Model Training Step:**
- Trains multiple machine learning models (e.g., Linear Regression, Random Forest, XGBoost).

#### **Model Evaluation Step:**
- Evaluates models based on metrics like RMSE, MAE, and R².
- Logs results for model comparison.

#### **Prediction Service Loader**
- Fetches and returns a running MLflow model deployment so it can be used to make decisions.

#### **Predictor**
- Sends input data to the deployed model to make decisions 

#### **Pipeline Orchestration:**
- The entire process is orchestrated using ZenML, ensuring each step is executed in the correct order.

---

## Running the Pipeline

### 1. Setting Up the Environment
```bash
pip install -r requirements.txt
```

### 2. Initialize ZenML
```bash
zenml init
```

### 3. Running the Pipeline
```bash
python run_pipeline.py
```

### 4. Visualizing the Pipeline
To track and visualize the pipeline execution:
```bash
zenml up --blocking
```
This will launch a ZenML dashboard where you can inspect pipeline runs, logs, and metrics.

## Exploratory Data Analysis (EDA)

Before building the pipeline, extensive EDA is conducted to understand the dataset:

- **Basic Data Inspection:** Checking dataset structure, data types, and summary statistics.
- **Missing Values Analysis:** Identifying and handling missing values.
- **Univariate Analysis:** Distribution of individual variables using histograms and boxplots.
- **Bivariate & Multivariate Analysis:** Correlation heatmaps and pairwise relationships between features.

---

## Technologies Used

- **Python**
- **ZenML** (for pipeline automation)
- **Pandas, NumPy** (for data processing)
- **Scikit-Learn** (for ML models)
- **Matplotlib, Seaborn** (for data visualization)

---

## Repository Structure

```
|── extracted_data/                   # Raw and processed data
|   |─── CarPrice_Assigment.csv
|── data/                   
|   |─── archive (1).zip
|── car-price-prediction/
  ├── analysis/
    ├── analyze_src/
    │   ├── basic_data_inspection.py        # Data ingestion script
    │   ├── bivariate_analysis.py    # Data preprocessing and feature engineering
    │   ├── missing_values_analysis.py            # Model training script
    │   ├── multivariate_analysis.py         # Model evaluation script
    │   ├── univariate_analysis.py         # ZenML pipeline definition
  ├── src/
  │   ├── data_ingest.py        
  │   ├── data_splitter.py    
  │   ├── feature_engineering.py            
  │   ├── handle_missing_values.py         
  │   ├── model_building.py
  │   ├── model_evaluator.py         
  │   ├── outlier_detection.py
  ├── steps/
  │   ├── data_ingesttion_step.py        
  │   ├── data_splitter_step.py    
  │   ├── feature_engineering_step.py            
  │   ├── handle_missing_values_step.py         
  │   ├── model_building_step.py
  │   ├── model_evaluator_step.py         
  │   ├── outlier_detection_step.py
  │   ├── model_loader.py         
  │   ├── prediction_service_loader.py
  │   ├── predictor.py         
  │   ├── training_pipeline.py
  ├── pipeline/
  │   ├── deployment_pipeline.py        
  │   ├── training_pipeline.py    
├── run_pipeline.py         # Main script to run the pipeline
├── run_deployment.py.py
├── sample_predict.py
├── requirements.txt        # Dependencies
```

---

## Conclusion

This project demonstrates the automation of an end-to-end data science workflow using ZenML. By structuring the pipeline into modular steps, we ensure reproducibility, scalability, and efficiency.

If you have any questions or suggestions, feel free to contribute or raise an issue in this repository. 🚀
