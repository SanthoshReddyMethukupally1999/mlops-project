import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train(model_path="model.joblib",C=1.00):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C))
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, model_path)
    return accuracy


if __name__ == "__main__":
    C = 0.01
    mlflow.set_experiment("breast-cancer")
    with mlflow.start_run():
        accuracy = train(C=C)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", accuracy)

    print(f"C={C}  accuracy={accuracy:.3f}")