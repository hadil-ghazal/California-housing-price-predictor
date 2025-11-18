#No AI was used to generate this code , written by HG on 11/17/25
#initial imports
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    #Loading dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={"MedHouseVal": "median_house_value"})

    #Features and target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    #Split data for train and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Linear Regression pipeline
    linear_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    linear_pipeline.fit(X_train, y_train)
    y_pred_lr = linear_pipeline.predict(X_test)
    print("Linear Regression R²:", round(r2_score(y_test, y_pred_lr), 4))
    print("Linear Regression MAE:", round(mean_absolute_error(y_test, y_pred_lr), 4))

    #Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest R²:", round(r2_score(y_test, y_pred_rf), 4))
    print("Random Forest MAE:", round(mean_absolute_error(y_test, y_pred_rf), 4))
    
    #Printing sample predictions for confirmation  few sample predictions
    sample_df = pd.DataFrame({
        "Actual (100k)": y_test.values[:5],
        "Predicted (100k)": y_pred_rf[:5]
    })
    
    print("\nSample Random Forest predictions(first 5 rows):")
    print(sample_df)


	
if __name__ == "__main__":
    main()

