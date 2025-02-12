from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(pipeline, X, y):

    # Expanded search space
    param_grid = {
        'gbr__n_estimators': [200, 300, 400],
        'gbr__learning_rate': [0.01, 0.05, 0.1],
        'gbr__max_depth': [3, 5, 7],
        'gbr__min_samples_split': [2, 4, 6]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_
