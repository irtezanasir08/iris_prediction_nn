### Iris Prediction Neural Network

This is a simple neural network starter project. It trains on four (conveniently) numeric features and predicts the species of an iris flower.

The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/uciml/iris).

The most interesting part is this code:
```python
history_multiclass = model.fit(
    X_train, y_train,
    epochs=50,       
    batch_size=10,  
    validation_split=0.1
)
```

Experimenting with the `epochs` and `batch_size` parameters will significantly affect the model's accuracy, especially since the dataset is quite small.