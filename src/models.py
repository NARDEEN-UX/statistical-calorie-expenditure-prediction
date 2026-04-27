import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

def build_simple_linear_regression(X_train, y_train, feature_name='Duration'):
    """
    Builds a simple linear regression model using a single dominant feature.
    """
    X_train_simple = X_train[[feature_name]]
    model = LinearRegression()
    model.fit(X_train_simple, y_train)
    print("--- SIMPLE LINEAR REGRESSION ---")
    print(f"Feature: {feature_name}")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    return model

def build_multivariable_regression_sm(X_train, y_train):
    """
    Builds a robust multivariable linear regression model using statsmodels.
    """
    X_train_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print("--- MULTIVARIABLE REGRESSION (STATSMODELS) ---")
    print(model_sm.summary())
    return model_sm

def build_multivariable_regression_sklearn(X_train, y_train):
    """
    Builds a multivariable linear regression model using scikit-learn.
    """
    model_sk = LinearRegression()
    model_sk.fit(X_train, y_train)
    print("--- MULTIVARIABLE REGRESSION (SCIKIT-LEARN) ---")
    print(f"Coefficients: {model_sk.coef_}")
    print(f"Intercept: {model_sk.intercept_}")
    return model_sk