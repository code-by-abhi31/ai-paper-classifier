import joblib
from sklearn import datasets

def main():
    # Load the trained model
    model = joblib.load("models/best_model.pkl")
    print("✅ Model loaded successfully!")

    # Load iris dataset again for testing
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Take one sample for inference
    sample = X[0].reshape(1, -1)  # first flower
    prediction = model.predict(sample)

    print(f"Sample features: {X[0]}")
    print(f"Predicted class: {iris.target_names[prediction][0]}")
    print(f"True class: {iris.target_names[y[0]]}")

if __name__ == "__main__":
    main()