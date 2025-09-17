import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# --- Import Random Forest Regressor ---
from sklearn.ensemble import RandomForestRegressor
# -------------------------------------
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings

# Suppress specific warnings if necessary
warnings.filterwarnings('ignore', category=UserWarning, message='.*feature_names mismatch.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. Load Data ---
try:
    # *** Make sure this points to your correct CSV file ***
    df = pd.read_csv('FINALIZED_DATASET.csv')
    print("Dataset loaded successfully.")
    df.columns = df.columns.str.strip()
    print("Column names stripped.")

    # --- Define ALL potential feature columns first for cleaning ---
    all_feature_columns = [
         'Size', 'Complexity', 'Priority',
        'Noftasks',  'developmenttype', 'platform', 'externalhardware',
        'relatedtechnologies', 'dbms', 'Requirement Volatility'
        'Teammembers', 'Experience'
        'PL',
    ]
    target_column = 'Effort'

    # --- Handle 'None' string meaning ---
    if 'relatedtechnologies' in df.columns:
        df['relatedtechnologies'].fillna('None', inplace=True)
    if 'dbms' in df.columns:
        df['dbms'].fillna('None', inplace=True)
    # --- End of Fix ---

except FileNotFoundError:
    print(f"Error: Dataset file not found. Please ensure the CSV file is named correctly and in the right directory.")
    exit()
except Exception as e:
    print(f"Error loading or processing dataset: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) - SELECTING BASE FEATURES ---
base_feature_columns = [
    'Size', 'Complexity', 'Priority',
        'Noftasks',  'developmenttype',  'externalhardware',
        'relatedtechnologies', 'dbms', 'Requirement Volatility',
        'Teammembers', 
        'PL'
]
print(f"\n--- Base features input to preprocessor: {base_feature_columns} ---")

# Check if base columns exist
missing_base_features = [col for col in base_feature_columns if col not in df.columns]
if missing_base_features:
    print(f"Error: One or more base features not found in dataset: {missing_base_features}")
    exit()

X = df[base_feature_columns]
y = df[target_column]

# Basic checks for target
if target_column not in df.columns:
    print(f"Error: Missing target column '{target_column}'. Available: {df.columns.tolist()}")
    exit()
if y.isnull().any():
    print(f"Warning: Target column '{target_column}' contains missing values.")
    print("Dropping rows with missing target values...")
    original_len = len(df)
    df.dropna(subset=[target_column], inplace=True)
    X = df[base_feature_columns] # Re-select X after dropping rows
    y = df[target_column]       # Re-select y after dropping rows
    print(f"Dropped {original_len - len(df)} rows.")
    if df.empty:
        print("Error: No data left after dropping rows with missing target.")
        exit()


# --- 3. Identify Feature Types *within the selected base subset* ---
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

print(f"\nIdentified Numerical Columns (within base subset): {numerical_cols}")
print(f"Identified Categorical Columns (within base subset): {categorical_cols}")

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # Fixed random state for split
)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# --- 5. Define Preprocessing Steps ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_transformers = []
if numerical_cols:
    preprocessor_transformers.append(('num', numeric_transformer, numerical_cols))
if categorical_cols:
    preprocessor_transformers.append(('cat', categorical_transformer, categorical_cols))

if not preprocessor_transformers:
     print("Error: No numerical or categorical columns identified in the base feature subset.")
     exit()

preprocessor = ColumnTransformer(
    transformers=preprocessor_transformers,
    remainder='passthrough'
)

# --- Determine max features after preprocessing (needed for k range) ---
try:
    X_train_processed_check = preprocessor.fit_transform(X_train)
    max_features_after_ohe = X_train_processed_check.shape[1]
    print(f"\nTotal number of features after OneHotEncoding the base features: {max_features_after_ohe}")
    if max_features_after_ohe < 3:
         print("Warning: Fewer than 3 features available after preprocessing. Adjusting K range.")
except Exception as e:
    print(f"Error during preprocessing check: {e}")
    print("Could not determine max features after OHE. Assuming a default max for K.")
    max_features_after_ohe = 12

# --- 6. Iterate through different K values for Feature Selection ---
results_k = []
results_mae = []
results_mse = []
results_rmse = []

# Define the range for k (3 to 12, but not exceeding available features)
k_values = range(3, min(13, max_features_after_ohe + 1))

print(f"\nEvaluating Random Forest Regressor for K features from {min(k_values)} to {max(k_values)}...")

for k in k_values:
    print(f" Processing K = {k} features...")

    # Define Feature Selector with seeded MI calculation for reproducibility
    feature_selector = SelectKBest(
        score_func=lambda X, y: mutual_info_regression(X, y, random_state=42), # Seed MI
        k=k
    )

    # --- Define the Random Forest Model ---
    # Use random_state for reproducible results
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # ------------------------------------

    # Create the Full Pipeline for this k
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', feature_selector),
        ('regressor', rf_model) # Use Random Forest model here
    ])

    # Train the Model
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"  Error fitting pipeline for K={k}: {e}")
        continue

    # Make Predictions (Standard predict works fine for RandomForest)
    try:
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        print(f"  Error predicting for K={k}: {e}")
        continue

    # Evaluate the Model
    try:
        valid_test_indices = y_test.notna()
        if valid_test_indices.sum() == 0:
            print(f"  Skipping evaluation for K={k}: No valid test targets.")
            continue

        mae = mean_absolute_error(y_test[valid_test_indices], y_pred[valid_test_indices.values])
        mse = mean_squared_error(y_test[valid_test_indices], y_pred[valid_test_indices.values])
        rmse = np.sqrt(mse)

        results_k.append(k)
        results_mae.append(mae)
        results_mse.append(mse)
        results_rmse.append(rmse)

    except Exception as e:
        print(f"  Error evaluating for K={k}: {e}")
        continue

print("Evaluation complete.")

# --- 7. Display Results in Table Format ---
if not results_k:
    print("\nNo results were generated for Random Forest. Check for errors during processing.")
else:
    results_df = pd.DataFrame({
        'K (Features)': results_k,
        'MAE': results_mae,
        'MSE': results_mse,
        'RMSE': results_rmse
    }).set_index('K (Features)')

    # --- Updated Title ---
    print("\n--- Random Forest Model Performance for Different Numbers of Selected Features (K) ---")
    # -------------------
    print(results_df.round(2)) # Rounding to 2 decimal places
    print("-----------------------------------------------------------------------------------")

# --- Add note about reproducibility ---
print("\nNote: Random states have been set for train/test split, mutual information scoring,")
print("and the Random Forest Regressor itself to maximize reproducibility across platforms.")
print("If results still differ slightly, check library versions (scikit-learn, pandas, numpy).")