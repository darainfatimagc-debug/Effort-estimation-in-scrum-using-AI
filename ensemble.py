# ensemble_evaluator_only_xgb_workaround.py
# Trains ET, RF, LR, XGB using specified features,
# calculates ensemble performance on the test set WITHOUT saving pipelines.
# Includes manual prediction workaround for XGBoost.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Import Required Regressors ---
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
# ----------------------------------

import traceback
import warnings
import sklearn

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*feature_names mismatch.*')

print(f"Using scikit-learn version: {sklearn.__version__}")
print(f"Using xgboost version: {xgb.__version__}")
print("--- Ensemble Evaluation Only (ET, RF, LR, XGB) Script ---")

# --- Configuration ---
DATASET_PATH = "FINALIZED_DATASET.csv" # Ensure this is correct

# --- 1. Load Data & Initial Cleaning ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    df.columns = df.columns.str.strip()

    # Handle NaNs in specific categorical columns
    if 'relatedtechnologies' in df.columns:
        df['relatedtechnologies'].fillna('None', inplace=True)
    if 'dbms' in df.columns:
        df['dbms'].fillna('None', inplace=True)

except FileNotFoundError:
    print(f"FATAL Error: Dataset file '{DATASET_PATH}' not found.")
    exit()
except Exception as e:
    print(f"FATAL Error loading data: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---
base_features = [
    'Size', 'Complexity', 'Priority', 'Noftasks', 'developmenttype',
    'externalhardware', 'relatedtechnologies', 'dbms', 'Requirement Volatility',
    'Teammembers', 'PL'
]
target_column = 'Effort'

print(f"\nUsing these {len(base_features)} features for all models: {base_features}")
print(f"Target column: {target_column}")

# Check columns, map text, convert to numeric, handle NaNs (Same as before)
required_cols = base_features + [target_column]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols: print(f"FATAL Error: Missing columns: {missing_cols}"); exit()

print("\nApplying text-to-numeric mapping (if needed)...")
size_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Highest': 5}
complexity_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Highest': 5}
req_volatility_map = {'High': 1, 'Medium': 2, 'Low': 3}
cols_to_map = {'Size': size_map, 'Complexity': complexity_map, 'Requirement Volatility': req_volatility_map}
for col, mapping in cols_to_map.items():
    if col in df.columns and df[col].dtype == 'object':
         print(f" - Mapping column '{col}'")
         df[col] = df[col].astype(str).replace(mapping)

print("Converting columns to numeric...")
cols_to_convert_numeric = ['Size', 'Complexity', 'Priority', 'Noftasks', 'externalhardware', 'Requirement Volatility', 'Teammembers', target_column]
for col in cols_to_convert_numeric:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=required_cols, inplace=True)
print(f"Data shape after NaN drop: {df.shape}")
if df.empty: print("FATAL Error: No data left after cleaning."); exit()

X = df[base_features].copy()
y = df[target_column]

# --- 3. Identify Feature Types for Preprocessor ---
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nFinal Numerical columns for preprocessor: {numerical_cols}")
print(f"Final Categorical columns for preprocessor: {categorical_cols}")
defined_cols = set(numerical_cols + categorical_cols); base_set = set(base_features)
if defined_cols != base_set: print(f"FATAL Error: Feature type mismatch: {base_set - defined_cols}"); exit()

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# --- 5. Define Preprocessing Steps ---
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)], remainder='passthrough')
print("\nPreprocessor defined.")

# --- 6. Define Individual Models ---
print("Defining base models...")
et_model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lr_model = LinearRegression()
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)

# --- 7. Create Full Pipelines ---
print("Creating full pipelines...")
et_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', et_model)])
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf_model)])
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lr_model)])
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb_model)]) # We still create the full pipeline to fit the preprocessor and model together

# --- 8. Train All Pipelines ---
print("\n--- Training Pipelines ---")
pipelines_to_train = {
    "Extra Trees": et_pipeline,
    "Random Forest": rf_pipeline,
    "Linear Regression": lr_pipeline,
    "XGBoost": xgb_pipeline
}
try:
    for name, pipe in pipelines_to_train.items():
        print(f"Training {name} pipeline...")
        pipe.fit(X_train, y_train) # This fits both preprocessor and regressor within the pipeline
    print("--- Training Complete ---")
except Exception as e:
    print(f"FATAL Error during training: {e}"); print(traceback.format_exc()); exit()

# --- 9. Predictions on Test Set (with XGBoost Workaround) --- # MODIFIED #
print("\n--- Making Predictions on Test Set ---")
predictions = {}
prediction_possible = True
try:
    print("Generating predictions for ET, RF, LR using pipelines...")
    predictions['et'] = et_pipeline.predict(X_test)
    predictions['rf'] = rf_pipeline.predict(X_test)
    predictions['lr'] = lr_pipeline.predict(X_test)

    # --- XGBoost Manual Prediction Workaround ---
    print("Generating XGBoost prediction manually...")
    # 1. Get the fitted preprocessor from the *trained* xgb_pipeline
    fitted_preprocessor = xgb_pipeline.named_steps['preprocessor']
    # 2. Transform the test data
    X_test_transformed = fitted_preprocessor.transform(X_test)
    # 3. Get the fitted XGBoost model from the *trained* xgb_pipeline
    fitted_xgb_model = xgb_pipeline.named_steps['regressor']
    # 4. Predict using the fitted XGBoost model directly
    predictions['xgb'] = fitted_xgb_model.predict(X_test_transformed)
    # --- End XGBoost Workaround ---

    print("Predictions generated successfully.")
except Exception as e:
    print(f"Error during prediction: {e}")
    print(traceback.format_exc())
    prediction_possible = False

# --- 10. Simple Average Ensemble Prediction ---
if prediction_possible:
    print("\nCalculating Ensemble prediction...")
    ensemble_pred = (predictions['et'] + predictions['rf'] + predictions['lr'] + predictions['xgb']) / 4
    print("Ensemble prediction calculated (Simple Average).")

    # --- 11. Evaluation of the Ensemble ---
    print("\n--- Simple Average Ensemble Model Evaluation (ET, RF, LR, XGB) ---")
    try:
        mae = mean_absolute_error(y_test, ensemble_pred)
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)

        print(f"Base Models: ExtraTrees, RandomForest, LinearRegression, XGBoost")
        print(f"Features Used: {len(base_features)}")
        print(f"MAE : {mae:.2f}")
        print(f"MSE : {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
else:
    print("\nEnsemble evaluation skipped due to errors in prediction step.")

print("\n--- Script Finished (No models saved) ---")