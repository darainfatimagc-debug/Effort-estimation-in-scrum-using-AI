# train_and_save_model.py
import pandas as pd
import numpy as np
# train_test_split ab zaroori nahi agar final model pure data pe train karna hai
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import joblib # Models save karne ke liye
import os     # Directory banane ke liye
import warnings
warnings.filterwarnings('ignore')

print("--- Model Training aur Saving Shuru ---")

# --- Configuration ---
DATASET_PATH = "FINALIZED_DATASET.csv"
MODEL_DIR = "models" # Models save karne ka folder
# Har pipeline ke liye alag file name
ET_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "et_pipeline.joblib")
RF_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "rf_pipeline.joblib")
MLP_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "mlp_pipeline.joblib")
XGB_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "xgb_pipeline.joblib")

# Model directory banayein agar nahi hai
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Model files yahan save hongi: {MODEL_DIR}")

# 1. Data Load aur Clean karna
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' load ho gaya.")
except FileNotFoundError:
    print(f"Error: Dataset file '{DATASET_PATH}' nahi mili.")
    exit()

df.columns = df.columns.str.strip()
df['relatedtechnologies'].fillna('None', inplace=True)
df['dbms'].fillna('None', inplace=True)

# 2. Features aur Target Define Karna
# !! IMPORTANT !!: Yeh features wahi hone chahiye jo aapke HTML form se aa rahe hain
#                  aur model ko predict karne ke liye chahiye.
base_features = ['Size', 'Complexity', 'Requirement Volatility', 'relatedtechnologies', 'externalhardware']
target_column = 'Effort'

# Check karein ki saare zaroori columns dataset mein hain
required_cols = base_features + [target_column]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Yeh required columns dataset mein nahi hain: {missing_cols}")
    exit()

print(f"In features ka istemal hoga: {base_features}")
print(f"Target column hai: {target_column}")

df.dropna(subset=[target_column], inplace=True)
X = df[base_features]
y = df[target_column]

if X.empty:
    print("Error: Target column mein NaN hatane ke baad koi data nahi bacha.")
    exit()

# 3. Categorical aur Numerical Columns Pehchan'na
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
print(f"Pehchane gaye Numerical columns: {numerical_cols}")
print(f"Pehchane gaye Categorical columns: {categorical_cols}")

# 4. Preprocessing Pipeline (Same as before)
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)
print("Preprocessor define ho gaya.")

# 5. Models Define Karna (Same parameters use karein)
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # 100 estimators use karein
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001,
    max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=10,
    validation_fraction=0.1
)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
print("Models define ho gaye.")

# 6. Full Pipelines Define Karna (Preprocessor + Model)
et_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', et_model)])
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', rf_model)])
mlp_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', mlp_model)])
xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', xgb_model)])
print("Full pipelines define ho gaye.")

# 7. Sab Models ko Train Karna (Pure dataset par - X, y)
#    Final deploy hone wale model ke liye aksar pure data par train karte hain.
print("\n--- Models ko Pure Dataset par Train Karna Shuru ---")
print("Training Extra Trees...")
et_pipeline.fit(X, y)
print("Training Random Forest...")
rf_pipeline.fit(X, y)
print("Training MLP...")
mlp_pipeline.fit(X, y)
print("Training XGBoost...")
xgb_pipeline.fit(X, y)
print("--- Model Training Mukammal ---")

# 8. Zaroori Components Save Karna
print("\n--- Pipelines Save Karna Shuru ---")
# Har fitted pipeline ko save karein
joblib.dump(et_pipeline, ET_PIPELINE_FILENAME)
print(f"Extra Trees pipeline save ho gayi: {ET_PIPELINE_FILENAME}")
joblib.dump(rf_pipeline, RF_PIPELINE_FILENAME)
print(f"Random Forest pipeline save ho gayi: {RF_PIPELINE_FILENAME}")
joblib.dump(mlp_pipeline, MLP_PIPELINE_FILENAME)
print(f"MLP pipeline save ho gayi: {MLP_PIPELINE_FILENAME}")
joblib.dump(xgb_pipeline, XGB_PIPELINE_FILENAME)
print(f"XGBoost pipeline save ho gayi: {XGB_PIPELINE_FILENAME}")

print("\n--- Model Training aur Saving Kamyabi Se Mukammal ---")