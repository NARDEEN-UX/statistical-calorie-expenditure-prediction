from data_loader import load_the_data
from eda import (check_data_quality, define_variable_types, 
                 get_min_max_values, check_outliers, plot_feature_correlation)
from preprocessing import remove_outliers, preprocess_data
from models import build_simple_linear_regression, build_multivariable_regression_sm
from evaluation import test_model_accuracy, perform_residual_analysis

def main():
    # 1. Load Data
    path = "calories.csv" # Adjust to your local or Kaggle data path
    df = load_the_data(path)
    
    # 2. Exploratory Data Analysis
    check_data_quality(df)
    quant_cols, cat_cols, target = define_variable_types(df)
    get_min_max_values(df)
    
    outlier_indices = check_outliers(df, quant_cols + [target])
    plot_feature_correlation(df, quant_cols, target)
    
    # 3. Preprocessing
    df_clean = remove_outliers(df, outlier_indices)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df_clean, target_col=target)
    
    # 4. Modeling
    # Simple Linear Regression
    simple_model = build_simple_linear_regression(X_train, y_train, feature_name='Duration')
    
    # Multivariable Regression
    multi_model_sm = build_multivariable_regression_sm(X_train, y_train)
    
    # 5. Evaluation & Diagnostics
    test_model_accuracy(multi_model_sm, X_test, y_test)
    perform_residual_analysis(multi_model_sm)

if __name__ == "__main__":
    main()