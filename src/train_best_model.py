from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "SVM": SVC(kernel='linear'),
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=200)
    }

    best_acc = 0
    best_model_name = ""
    best_model = None

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model

    print(f"\n✅ Best Model: {best_model_name} with Accuracy: {best_acc:.4f}")

    # Save the best model
    joblib.dump(best_model, "models/best_model.pkl")
    print("✅ Model saved to models/best_model.pkl")

if __name__ == "__main__":
    main()