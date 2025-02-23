import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
        "car_ID": [5, 6],
        "symboling": [2, 2],
        "wheelbase": [99.4, 20],
        "carlength": [176.6, 177.3],
        "carwidth": [66.4, 66.3],
        "carheight": [54.3, 53.1],
        "curbweight": [2824, 2507],
        "enginesize": [136, 136],
        "boreratio": [3.19, 3.19],
        "stroke": [3.4, 3.4 ],
        "compressionratio": [8, 978],
        "horsepower": [115, 0],
        "peakrpm": [5500, 284],
        "citympg": [18, 1262],
        "highwaympg": [22, 1262]
    }

    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
