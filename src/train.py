import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from utils.read import Read
from utils.constants import Constants
from pandas import DataFrame
from mlflow.models.signature import infer_signature

# Usar tracking local en lugar de servidor remoto
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("adult-classification-hpo-bestmodel")


def run_optimization(df: DataFrame):

    X = df.drop('income', axis = 1) # Variables predictoras
    Y = df['income'] #Variable objetivo
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, stratify=Y)
    
    best_model = None
    best_accuracy = 0
    best_params = None
    best_model_name = None

    def objective(trial):
        nonlocal best_model, best_accuracy, best_params, best_model_name
        
        with mlflow.start_run():
            model_name = trial.suggest_categorical('model', ['RandomForest', 'LogisticRegression', 'SVM'])
            
            if model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            
            elif model_name == 'LogisticRegression':
                C_lr = trial.suggest_float('C_lr', 0.01, 10.0, log=True)
                params = {
                    'C': C_lr,
                    'max_iter': 2000,
                    'random_state': 42
                }
                model = LogisticRegression(**params)
            
            else:  # SVM
                C_svm = trial.suggest_float('C_svm', 0.1, 10.0)
                params = {
                    'C': C_svm,
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                    'random_state': 42
                }
                model = SVC(**params)

            mlflow.log_param('model_type', model_name)
            mlflow.log_params(params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            print(f"Trial {trial.number}: {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Actualizar mejor modelo
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_params = params
                best_model_name = model_name
            
            return accuracy

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20)
    
    # Guardar solo el mejor modelo
    with mlflow.start_run():
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.log_param('model_type', best_model_name)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.sklearn.log_model(
        best_model, 
        "best_model",
        signature=signature
    )
        mlflow.set_tag("status", "best_model")
        mlflow.set_tags({"version": "1.0.0",
                        "developer": "Sebastian y Juan Manuel",
                        "dataset": "adult"})
        print(f"\nBest model saved: {best_model_name} with accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    root_path = Constants.root_path.value
    procesed_path = Constants.path_processed.value
    processed_file = Constants.processed_file.value
    df = Read.read_parquet(f'{root_path}{procesed_path}{processed_file}.parquet')
    run_optimization(df)
