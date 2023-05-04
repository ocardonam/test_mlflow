import mlflow

import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])

    with mlflow.start_run() as run:
        # Load the diabetes dataset.
        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

        # Create and train models.
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=3)
        rf.fit(X_train, y_train)

        # Use the model to make predictions on the test dataset.
        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        print(mse)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_param("N_estimators", n_estimators)
        mlflow.log_param("Max depth", max_depth)
        mlflow.sklearn.log_model(rf, "model")
        

        print("Run ID: {}".format(run.info.run_id))