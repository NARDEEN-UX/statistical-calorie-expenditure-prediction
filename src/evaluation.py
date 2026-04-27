import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np

def test_model_accuracy(model_sm, X_test, y_test):
    """
    Tests model accuracy on the test set and plots actual vs predicted values.
    """
    X_test_sm = sm.add_constant(X_test)
    y_pred = model_sm.predict(X_test_sm)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("--- TEST SET PERFORMANCE ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Actual vs. Predicted Calories (Test Set)')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.legend(['Perfect Prediction', 'Predictions'])
    plt.grid(True)
    plt.show()

def perform_residual_analysis(model_sm):
    """
    Performs rigorous normality tests on the residuals.
    """
    residuals = model_sm.resid
    
    # 1. Q-Q Plot
    sm.qqplot(residuals, line='45', fit=True)
    plt.title('Q-Q Plot of Residuals')
    plt.show()
    
    # 2. Anderson-Darling Test
    ad_stat, ad_critical, ad_sig = stats.anderson(residuals, dist='norm')
    print("\n--- ANDERSON-DARLING TEST ---")
    print(f"Statistic: {ad_stat:.4f}")
    for i in range(len(ad_critical)):
        print(f"Significance {ad_sig[i]}%: Critical Value = {ad_critical[i]:.3f}")
    
    # 3. D'Agostino-Pearson Test
    dp_stat, dp_pvalue = stats.normaltest(residuals)
    print("\n--- D'AGOSTINO-PEARSON TEST ---")
    print(f"Statistic: {dp_stat:.4f}, p-value: {dp_pvalue:.4e}")