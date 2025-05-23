import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

def main():
    # Set nama eksperimen (akan otomatis dibuat lokal di mlruns/)
    mlflow.set_experiment("Obesity_Classification")

    # Aktifkan autologging
    mlflow.sklearn.autolog(log_models=False)

    # Load dataset
    df = pd.read_csv("obesity_data_preprocessing.csv")
    X = df.drop("ObesityCategory", axis=1)
    y = df["ObesityCategory"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)

    with mlflow.start_run(run_name="Random Forest with Tuning"):
        grid_search.fit(X_train, y_train)
        best_rf_model = grid_search.best_estimator_

        y_pred = best_rf_model.predict(X_test)

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        f1_macro = report_dict["macro avg"]["f1-score"]
        class_0_f1 = report_dict.get("0", {}).get("f1-score", 0)
        num_classes = len(report_dict) - 3
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        fp = cm[0][1]
        specificity = tn / (tn + fp)
        f1_class_1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_class_0", class_0_f1)
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("f1_class_1", f1_class_1)

        input_example = X_train.iloc[:1].astype("float64")
        signature = infer_signature(X_train.astype("float64"), best_rf_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_rf_model,
            artifact_path="random_forest_model",
            input_example=input_example,
            signature=signature
        )

        print("✅ Model logged to MLflow.")

if __name__ == "__main__":
    main()
