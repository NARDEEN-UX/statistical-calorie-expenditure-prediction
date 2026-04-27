import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def remove_outliers(df, outlier_indices):
    """
    Removes the rows containing outliers from the dataset.
    """
    print(f"Removing {len(outlier_indices)} outliers...")
    return df.drop(index=list(outlier_indices)).reset_index(drop=True)

def preprocess_data(df, target_col='Calories'):
    """
    Handles data splitting, one-hot encoding, and standardization.
    """
    # Drop User_ID as it is merely an identifier
    if 'User_ID' in df.columns:
        df = df.drop(columns=['User_ID'])
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # One-hot encoding for categorical variables (Gender)
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Standardization
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    cols_to_scale = [col for col in numerical_cols if col not in ['Gender_male', 'Gender_female']]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler