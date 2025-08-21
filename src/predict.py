import joblib
from sklearn.datasets import load_iris

def main():
    # Load scaler + model
    scaler, model = joblib.load("../models/best_model.pkl")
    
    # Load iris to get target names
    iris = load_iris()
    target_names = iris.target_names

    # Example input (sepal length, sepal width, petal length, petal width)
    sample = [[5.1, 3.5, 1.4, 0.2]]

    # Scale input
    scaled_sample = scaler.transform(sample)

    # Predict
    prediction = model.predict(scaled_sample)[0]
    print("Predicted class index:", prediction)
    print("Predicted class name:", target_names[prediction])

if __name__ == "__main__":
    main()
