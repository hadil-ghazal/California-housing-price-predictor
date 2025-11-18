## Results

From the final run:


Linear Regression: R² = 0.5758, MAE = 0.5332
Random Forest:     R² = 0.8040, MAE = 0.3282
# California-housing-price-predictor
An end to end ML pipeline for predicting California housing prices using the CA Housing dataset. Includes preprocessing, two baseline models (Linear Regression and Random Forest), StandardScaler, and final train/test evaluation

R2 measures how much of the variance in the house proces the model explains, with higher being better
Mean absolute error measures teh average difference between predicted and the actual prices, with lower being better here

The random Forest Model performed significantly better than the simple linear regression which was baseline


## How to run
This project only needs Python and the packages listed below to recreate the results

### Step 1. Clone the repository

On bash : 
git clone https://github.com/<your-username>/california-housing-price-predictor.git
cd california-housing-price-predictor

### Step 2. Install dependencies 

On bash:
pip install -r requirements.txt

### Step 3: Rerun the training script
on bash:
python code/train.py

### Open notebook using jupyter notebook notebooks/california_housing_pipeline.ipynb


Expanded results:
Linear Regression R²: 0.5758
Linear Regression MAE: 0.5332
Random Forest R²: 0.8040
Random Forest MAE: 0.3282
 
