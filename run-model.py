import argparse
import json
from pathlib import Path
import numpy as np
import joblib
import sys

MODEL_PATH = Path("artifacts/model.pkl")

def load_model():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

def parse_input(input_arg):
    """
    Supports:
    1. JSON file: {"features": [..]}
    2. Raw CLI values: 10,1,5,10
    """

    input_path = Path(input_arg)

    # Case 1: JSON file
    if input_path.exists():
        with open(input_path, "r") as f:
            data = json.load(f)

        if "features" not in data:
            raise ValueError("JSON must contain a 'features' key")

        values = data["features"]

    # Case 2: Raw CLI input
    else:
        try:
            values = [float(x) for x in input_arg.split(",")]
        except ValueError:
            raise ValueError("Raw input must be comma-separated numeric values")

    if len(values) != 4:
        raise ValueError("Input must contain exactly 4 feature values")

    return np.array(values).reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="Run the trained Iris model")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file OR comma-separated values (e.g. 5.1,3.5,1.4,0.2)"
    )
    args = parser.parse_args()

    try:
        model = load_model()
        features = parse_input(args.input)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        result = {
            "input": features.tolist()[0],
            "predicted_class": int(prediction),
            "prediction_probabilities": probabilities.tolist()
        }

        print("Prediction result:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
