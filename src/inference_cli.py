import argparse
import joblib
import numpy as np
from sklearn import datasets

def main():
    # Load trained model
    model = joblib.load("models/best_model.pkl")

    # Load iris dataset to get class names
    iris = datasets.load_iris()
    class_names = iris.target_names

    # CLI arguments
    parser = argparse.ArgumentParser(description="Iris Flower Classifier")
    parser.add_argument("--sepal_length", type=float, required=True, help="Sepal length in cm")
    parser.add_argument("--sepal_width", type=float, required=True, help="Sepal width in cm")
    parser.add_argument("--petal_length", type=float, required=True, help="Petal length in cm")
    parser.add_argument("--petal_width", type=float, required=True, help="Petal width in cm")

    args = parser.parse_args()

    # Prepare input
    sample = np.array([[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]])

    # Predict
    prediction = model.predict(sample)[0]
    print(f"🌸 Predicted class: {class_names[prediction]}")

if __name__ == "__main__":
    main()