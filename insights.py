def generate_insights(df, model_pipeline):

    feature_cols = [
        'monthly_revenue', 'customer_visits', 'marketing_spend',
        'churn_risk_score', 'revenue_per_visit', 'marketing_efficiency'
    ]
    
    X = df[feature_cols]
    predicted_ltv = model_pipeline.predict(X)
    
    # Thresholds for logic
    revenue_threshold = df['monthly_revenue'].quantile(0.75)
    high_ltv_threshold = df['customer_lifetime_value'].quantile(0.75)
    
    insights = []
    
    for i, row in df.iterrows():
        # High churn risk
        if row['churn_risk_score'] > 0.7:
            insights.append(
                f"Customer {row['customer_id']} is at high churn risk (score: {row['churn_risk_score']:.2f}). "
                "Offer a discount or loyalty program."
            )
        # Loyal high-value
        elif (row['churn_risk_score'] < 0.3) and (row['monthly_revenue'] > revenue_threshold):
            insights.append(
                f"Customer {row['customer_id']} seems loyal and high-value (revenue: {row['monthly_revenue']}). "
                "Consider offering exclusive rewards or VIP perks."
            )
        
        # Low spending upsell
        if row['monthly_revenue'] < 1000:
            insights.append(
                f"Customer {row['customer_id']} has low spending (revenue: {row['monthly_revenue']}). "
                "Recommend upselling premium products or bundles."
            )
        
        # Declining LTV check (predicted vs. actual)
        if predicted_ltv[i] < row['customer_lifetime_value'] * 0.85:
            insights.append(
                f"Customer {row['customer_id']} shows a potential decline in LTV. "
                "Consider retention strategies."
            )
        
        # High LTV
        if row['customer_lifetime_value'] > high_ltv_threshold:
            insights.append(
                f"Customer {row['customer_id']} has a high lifetime value (CLTV: {row['customer_lifetime_value']:.2f}). "
                "Offer top-tier loyalty benefits or VIP experiences."
            )
    
    return insights
