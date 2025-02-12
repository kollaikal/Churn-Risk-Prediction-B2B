import numpy as np
import pandas as pd

def simulate_business_data(n_samples=300):
    """
    Simulates business data with a more predictable relationship
    so the model can achieve lower MAE.
    """
    np.random.seed(42)
    
    customer_id = np.arange(1, n_samples+1)
    
    # Core features
    monthly_revenue = np.random.randint(500, 5000, n_samples)
    customer_visits = np.random.randint(1, 30, n_samples)
    churn_risk_score = np.random.uniform(0, 1, n_samples)
    marketing_spend = np.random.randint(100, 2000, n_samples)

    # Engineered features
    revenue_per_visit = monthly_revenue / customer_visits
    marketing_efficiency = monthly_revenue / (marketing_spend + 1)  # Avoid division-by-zero

    # We'll create a synthetic relationship that is somewhat realistic:
    #   LTV ~ (12 months * monthly_revenue) modified by churn risk & visits
    #   Then add some moderate random noise.
    
    # Base LTV influenced by monthly revenue & churn risk
    base_ltv = (monthly_revenue * 12) * (1.5 - churn_risk_score)
    
    # Add a small premium for customers with more visits
    visit_influence = np.log1p(customer_visits) * 100
    
    # Random noise term
    noise = np.random.normal(loc=0, scale=2000, size=n_samples)
    
    customer_lifetime_value = base_ltv + visit_influence + noise

    #creating dataframe
    df = pd.DataFrame({
        'customer_id': customer_id,
        'monthly_revenue': monthly_revenue,
        'customer_visits': customer_visits,
        'churn_risk_score': churn_risk_score,
        'marketing_spend': marketing_spend,
        'customer_lifetime_value': customer_lifetime_value
    })
    
    # Additional Engineered Features
    df['revenue_per_visit'] = revenue_per_visit
    df['marketing_efficiency'] = marketing_efficiency
    df['churn_indicator'] = df['churn_risk_score'].apply(lambda x: 1 if x > 0.5 else 0)
    
    # Handle potential NaNs (should be none, but just in case)
    df.fillna(0, inplace=True)
    
    return df
