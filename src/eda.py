import matplotlib.pyplot as plt
import seaborn as sns

def check_data_quality(df):
    """
    Checks the dataframe for Missing, Zero, and Negative values.
    """
    print("--- 2. DATA QUALITY CHECK ---")
    numerical_cols = df.select_dtypes(include=['number']).columns
    print(f"{'Column Name':<20} | {'Missing':<8} | {'Zeros':<8} | {'Negatives':<10}")
    print("-" * 55)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if col in numerical_cols:
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
        else:
            zero_count = "N/A"
            negative_count = "N/A"
        print(f"{col:<20} | {missing_count:<8} | {zero_count:<8} | {negative_count:<10}")

def define_variable_types(df, target_col='Calories'):
    """
    Identifies quantitative and categorical predictor variables.
    """
    quantitative_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in ['User_ID', target_col]:
        if col in quantitative_cols:
            quantitative_cols.remove(col)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return quantitative_cols, categorical_cols, target_col

def get_min_max_values(df):
    """
    Prints the Minimum and Maximum values for every numerical column.
    """
    print("--- MIN / MAX CHECK ---")
    print(f"{'Column Name':<20} | {'Min Value':<12} | {'Max Value':<12}")
    print("-" * 50)
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        print(f"{col:<20} | {df[col].min():<12} | {df[col].max():<12}")

def check_outliers(df, quantitative_cols):
    """
    Checks for outliers in quantitative columns using the IQR Method (1.5 * IQR).
    """
    print("--- OUTLIER DETECTION (IQR Method) ---")
    outlier_indices = set()
    for col in quantitative_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        current_outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
        outlier_indices.update(current_outliers.index)
        print(f"{col}: Found {len(current_outliers)} outliers.")
    
    return outlier_indices

def plot_feature_correlation(df, quantitative_cols, target_col):
    """
    Plots the correlation matrix heatmap.
    """
    cols = quantitative_cols + [target_col]
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()