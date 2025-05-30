import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

def load_csv_for_tensorflow(file_path, feature_columns, label_column, test_size=0.1, shuffle_train=True, batch_size=32, random_state=42):
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)

        # Handle missing values
        for col in feature_columns:
            if df[col].dtype in [np.float64, np.int64, float, int]: 
                df[col].fillna(df[col].mean(), inplace=True)
            else: 
                df[col].fillna(df[col].mode()[0], inplace=True) 

        # Make sure label column has no nans 
        df.dropna(subset=[label_column], inplace=True)

        X = df[feature_columns]
        y = df[label_column]

        # Convert string labels to ints 
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_) 

        X_np = X.astype(np.float32).values
        y_np = y_encoded.astype(np.int32)

        # Reshape y_np to be a 2D array 
        if len(y_np.shape) == 1:
            y_np = y_np.reshape(-1, 1)

        # Split data into training and testing
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_np, y_np, test_size=test_size, random_state=random_state, shuffle=True, stratify=y_np
        )

        # print the shaopes
        print(f"Original features shape: {X_np.shape}, Original labels shape: {y_np.shape}")
        print(f"Training features shape: {X_train_np.shape}, Training labels shape: {y_train_np.shape}")
        print(f"Testing features shape: {X_test_np.shape}, Testing labels shape: {y_test_np.shape}")
        print(f"Number of unique classes: {num_classes}")


        return X_train_np, y_train_np, X_test_np, y_test_np, num_classes

    except FileNotFoundError:
        print(f"Error: File '{file_path}' was not found.")
        return None, None, None, None, None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None, None

feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# Make sure to update the filepath.
X_train, y_train, X_test, y_test, num_classes = load_csv_for_tensorflow(
    '/Users/irteza/Desktop/datasets/iris.csv', feature_columns, 'species'
)

if X_train is None:
    print("Exiting due to an error in loading data.")
else:
    print(f"\nShape of training data (X_train): {X_train.shape}")
    print(f"Shape of training labels (y_train): {y_train.shape}")
    print(f"Shape of testing data (X_test): {X_test.shape}")
    print(f"Shape of testing labels (y_test): {y_test.shape}")
    print(f"Number of classes: {num_classes}")

    model = tf.keras.models.Sequential([
        ## hidden layer 1: 8 nodes; relu activation
        tf.keras.layers.Dense(units=8, activation='relu', input_shape=(X_train.shape[1],)),
        ## hidden 2: 4 nodes; relu activation
        tf.keras.layers.Dense(units=4, activation='relu'),
        ## output layer: 'num_classes' nodes; softmax activation
        tf.keras.layers.Dense(units=num_classes, activation='softmax') # Use num_classes for the output layer
    ])

    model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
                             metrics=['accuracy'])

    model.summary()

    print("\n Training Model..")
    history_multiclass = model.fit(
        X_train, y_train,
        epochs=50,       
        batch_size=10,  
        validation_split=0.1
    )

    # Run evals
    print("\n--- Evaluating Multi-Class Model ---")
    loss_mc, accuracy_mc = model.evaluate(X_test, y_test, verbose=2) # verbose=2 prints one line per epoch
    print(f"Test Loss (Multi-Class): {loss_mc:.4f}")
    print(f"Test Accuracy (Multi-Class): {accuracy_mc:.4f}")
    