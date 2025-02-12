model:
  n_estimators: [200, 300, 400]
  learning_rate: [0.01, 0.05, 0.1]
  max_depth: [3, 5, 7]
  min_samples_split: [2, 4, 6]

data:
  seed: 42
  train_size: 0.8
