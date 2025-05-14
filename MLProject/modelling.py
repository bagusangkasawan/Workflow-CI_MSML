import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

mlflow.set_experiment("Obesity_Classification")

BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, "obesity_data_preprocessing.csv"))
X = df.drop("ObesityCategory", axis=1)
y = df["ObesityCategory"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
report_dict = classification_report(y_test, y_pred, output_dict=True)
f1_macro = report_dict["macro avg"]["f1-score"]
class_0_f1 = report_dict.get("0", {}).get("f1-score", 0)
num_classes = len(report_dict) - 3
cm = confusion_matrix(y_test, y_pred)
tn = cm[0][0]
fp = cm[0][1]
specificity = tn / (tn + fp)
f1_class_1 = f1_score(y_test, y_pred, average='macro')
    
with mlflow.start_run(run_name="Random Forest with Tuning") as run:
    run_id = run.info.run_id
    artifacts_path = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifacts_path, exist_ok=True)
    with open(os.path.join(artifacts_path, "run_id.txt"), "w") as f:
        f.write(run.info.run_id)
    
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("num_classes", num_classes)
    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("f1_class_0", class_0_f1)
    mlflow.log_metric("specificity", specificity)
    mlflow.log_metric("f1_class_1", f1_class_1)

    mlflow.sklearn.log_model(best_rf_model, artifact_path="random_forest_model")
    print("✅ Model logged to MLflow.")
