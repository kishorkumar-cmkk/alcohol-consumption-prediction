import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib

data = pd.read_csv("beer-servings-cleaned.csv")


features = [
    "country",
    "beer_servings",
    "spirit_servings",
    "wine_servings",
    "continent"
]

x = data[features]

y = data["total_litres_of_pure_alcohol"]



numerical_features = [
    "beer_servings",
    "spirit_servings",
    "wine_servings"
]

categorical_features = [
    "country",
    "continent"
]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

## Linear Regression

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

linear_model.fit(x_train, y_train)

linear_pred = linear_model.predict(x_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)



## Random Forest Regressor

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(x_train, y_train)

rf_pred = random_forest_model.predict(x_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)



## Gradient Boosting Regressor

gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                random_state=42
            )
        )
    ]
)

gradient_boosting_model.fit(x_train, y_train)

gb_pred = gradient_boosting_model.predict(x_test)

gb_r2 = r2_score(y_test, gb_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_rmse = np.sqrt(gb_mse)



results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],

    "R2 Score": [
        linear_r2,
        rf_r2,
        gb_r2
    ],

    "MAE": [
        linear_mae,
        rf_mae,
        gb_mae
    ],

    "MSE": [
        linear_mse,
        rf_mse,
        gb_mse
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse,
        gb_rmse
    ]
})

results = results.sort_values(
    by="R2 Score",
    ascending=False
)

print("\nModel Comparison")
print(results)

## Hyperparameter Tuning
## Tune Random Forest

rf_param_grid = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [None, 5, 10, 20],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4]
}

rf_grid_search = GridSearchCV(
    estimator=random_forest_model,
    param_grid=rf_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nTuning Random Forest...")

rf_grid_search.fit(x_train, y_train)

print("\nBest Random Forest Parameters:")
print(rf_grid_search.best_params_)

print("\nBest Random Forest CV R2:")
print(rf_grid_search.best_score_)

best_rf_model = rf_grid_search.best_estimator_
tuned_rf_pred = best_rf_model.predict(x_test)

tuned_rf_r2 = r2_score(y_test, tuned_rf_pred)
tuned_rf_mae = mean_absolute_error(y_test, tuned_rf_pred)
tuned_rf_mse = mean_squared_error(y_test, tuned_rf_pred)
tuned_rf_rmse = np.sqrt(tuned_rf_mse)

print("\nTuned Random Forest")
print("R2 Score:", tuned_rf_r2)
print("MAE:", tuned_rf_mae)
print("MSE:", tuned_rf_mse)
print("RMSE:", tuned_rf_rmse)



## Tune Gradient Boosting

gb_param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__max_depth": [2, 3, 4],
    "regressor__min_samples_split": [2, 5, 10]
}

gb_grid_search = GridSearchCV(
    estimator=gradient_boosting_model,
    param_grid=gb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nTuning Gradient Boosting...")

gb_grid_search.fit(x_train, y_train)

print("\nBest Gradient Boosting Parameters:")
print(gb_grid_search.best_params_)

print("\nBest Gradient Boosting CV R2:")
print(gb_grid_search.best_score_)

best_gb_model = gb_grid_search.best_estimator_

tuned_gb_pred = best_gb_model.predict(x_test)

tuned_gb_r2 = r2_score(y_test, tuned_gb_pred)
tuned_gb_mae = mean_absolute_error(y_test, tuned_gb_pred)
tuned_gb_mse = mean_squared_error(y_test, tuned_gb_pred)
tuned_gb_rmse = np.sqrt(tuned_gb_mse)

print("\nTuned Gradient Boosting")
print("R2 Score:", tuned_gb_r2)
print("MAE:", tuned_gb_mae)
print("MSE:", tuned_gb_mse)
print("RMSE:", tuned_gb_rmse)

final_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Tuned Random Forest",
        "Tuned Gradient Boosting"
    ],

    "R2 Score": [
        linear_r2,
        tuned_rf_r2,
        tuned_gb_r2
    ],

    "MAE": [
        linear_mae,
        tuned_rf_mae,
        tuned_gb_mae
    ],

    "MSE": [
        linear_mse,
        tuned_rf_mse,
        tuned_gb_mse
    ],

    "RMSE": [
        linear_rmse,
        tuned_rf_rmse,
        tuned_gb_rmse
    ]
})

final_results = final_results.sort_values(
    by="R2 Score",
    ascending=False
)

print("\nFinal Model Comparison")
print(final_results)

linear_model
best_model = linear_model

joblib.dump(best_model, "best_model.pkl")

print("\nBest Model: Linear Regression")
print("Final R2 Score:", linear_r2)
print("Model saved successfully as best_model.pkl")

