from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)

def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="C:/Users/LAKSHYA PALIWAL/Projects/car-price-prediction/data/archive (1).zip"
    )

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        raw_data, strategy="log", features=["enginesize", "price"]
    )

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data, column_name="price")

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="price")

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()