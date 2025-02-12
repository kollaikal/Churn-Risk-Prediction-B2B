import shap

def explain_model_with_shap(model_pipeline, X_sample):

    scaler = model_pipeline.named_steps['scaler']
    gbr_model = model_pipeline.named_steps['gbr']
    
    # Transform sample features
    X_transformed = scaler.transform(X_sample)
    
    # Create SHAP explainer
    shap_explainer = shap.Explainer(gbr_model, X_transformed)
    shap_values = shap_explainer(X_transformed)
    
    return shap_values, shap_explainer
