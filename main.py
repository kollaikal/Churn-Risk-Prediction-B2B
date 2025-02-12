import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# imports from your modular files
from simulate_data import simulate_business_data
from model import build_model_pipeline
from hyperparameter_tuning import tune_hyperparameters
from insights import generate_insights
from explain import explain_model_with_shap

def main():
    df = simulate_business_data(n_samples=300)

    # 2. Defining features and target
    feature_cols = [
        'monthly_revenue', 'customer_visits', 'marketing_spend',
        'churn_risk_score', 'revenue_per_visit', 'marketing_efficiency'
    ]
    target_col = 'customer_lifetime_value'
    
    X = df[feature_cols]
    y = df[target_col]

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Build pipeline
    pipeline = build_model_pipeline()

    # 5. Hyperparameter tuning
    print("Tuning hyperparameters. This may take a moment...")
    best_model_pipeline, best_params = tune_hyperparameters(pipeline, X_train, y_train)
    
    print("Best Hyperparameters found:", best_params)

    # 6. Train final model
    best_model_pipeline.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = best_model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance on Test Set:")
    print(f"- MAE: {mae:.2f}")
    print(f"- R^2: {r2:.2f}")

    # 8. Generate sample insights
    insights = generate_insights(df, best_model_pipeline)
    print("\nSample Insights:")
    for i in insights[:10]:
        print("-", i)

    # 9. SHAP Explanations
    try:
        print("\nGenerating SHAP explanations for the first 5 samples in test set...")
        X_test_sample = X_test.iloc[:5]
        shap_values, shap_explainer = explain_model_with_shap(best_model_pipeline, X_test_sample)
        print("SHAP values computed. You can plot them using 'shap.plots.bar(shap_values[i])' or 'shap.plots.waterfall(shap_values[i])'.")
    except Exception as e:
        print("SHAP explanation skipped due to error:", e)


if __name__ == "__main__":
    main()
