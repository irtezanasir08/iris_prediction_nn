import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

def load_csv_for_tensorflow(file_path, feature_columns, label_column, test_size=0.1, shuffle_train=True, batch_size=32, random_state=42):
    """
    Reads a CSV file, processes it, splits into training and testing sets,
    and prepares data structures suitable for TensorFlow neural network training.
    Specifically handles string labels by encoding them to integers.

    Args:
        file_path (str): Path to the CSV file.
        feature_columns (list of str): List of column names to be used as features.
        label_column (str): Name of the column to be used as the label.
        test_size (float): Proportion of the dataset to include in the test split (e.g., 0.1 for 10%).
        shuffle_train (bool): Whether to shuffle the training dataset.
        batch_size (int): Batch size for the tf.data.Dataset.
        random_state (int, optional): Seed for random number generation for reproducible splits.

    Returns:
        tuple: A tuple containing:
            - X_train_np (np.ndarray): NumPy array of training features.
            - y_train_np (np.ndarray): NumPy array of training labels (integer encoded).
            - X_test_np (np.ndarray): NumPy array of testing features.
            - y_test_np (np.ndarray): NumPy array of testing labels (integer encoded).
            - num_classes (int): The number of unique classes in the label column.
            Returns (None, ..., None) if an error occurs.
    """
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)

        # --- Data Preprocessing (Basic Example) ---
        # 1. Handle missing values
        for col in feature_columns:
            if df[col].dtype in [np.float64, np.int64, float, int]: # Check if column is numeric
                df[col].fillna(df[col].mean(), inplace=True)
            else: # For non-numeric (e.g., object, string)
                df[col].fillna(df[col].mode()[0], inplace=True) # Fill with mode

        # Ensure label column has no NaNs (e.g., drop rows with NaN labels)
        df.dropna(subset=[label_column], inplace=True)

        # Select features and labels
        X = df[feature_columns]
        y = df[label_column]

        # Convert string labels to numerical integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_) # Get the number of unique classes

        # Convert to NumPy arrays with a common data type (e.g., float32 for features)
        X_np = X.astype(np.float32).values
        y_np = y_encoded.astype(np.int32) # Labels should be integers for sparse_categorical_crossentropy

        # Reshape y_np to be a 2D array if it's not already (though LabelEncoder typically returns 1D)
        if len(y_np.shape) == 1:
            y_np = y_np.reshape(-1, 1)

        # --- Split data into training and testing sets ---
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_np, y_np, test_size=test_size, random_state=random_state, shuffle=True, stratify=y_np
        )
        # Added stratify=y_np to ensure that the proportion of target classes is the same in both the training and testing sets. This is good practice for classification tasks, especially with imbalanced datasets.

        print(f"Original features shape: {X_np.shape}, Original labels shape: {y_np.shape}")
        print(f"Training features shape: {X_train_np.shape}, Training labels shape: {y_train_np.shape}")
        print(f"Testing features shape: {X_test_np.shape}, Testing labels shape: {y_test_np.shape}")
        print(f"Number of unique classes: {num_classes}")


        return X_train_np, y_train_np, X_test_np, y_test_np, num_classes

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None
    except KeyError as e:
        print(f"Error: Column not found in CSV - {e}. Check feature_columns and label_column names.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None, None

# --- Example Usage ---
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# Make sure the path to your iris.csv is correct.
# For example, if it's in the same directory as your script, just 'iris.csv' might work.
X_train, y_train, X_test, y_test, num_classes = load_csv_for_tensorflow(
    '/Users/irteza/Desktop/datasets/iris.csv', feature_columns, 'species'
)

# Check if data loading was successful
if X_train is None:
    print("Exiting due to an error in loading data.")
else:
    print(f"\nShape of training data (X_train): {X_train.shape}")
    print(f"Shape of training labels (y_train): {y_train.shape}")
    print(f"Shape of testing data (X_test): {X_test.shape}")
    print(f"Shape of testing labels (y_test): {y_test.shape}")
    print(f"Number of classes: {num_classes}")

    # We'll use a simple Sequential model.
    # Input layer: matches the number of features in our data (4 for iris)
    # Hidden layers: a few neurons with ReLU activation
    # Output layer: 'num_classes' neurons with softmax activation for multi-class classification

    model = tf.keras.models.Sequential([
        ## hidden layer 1: 8 nodes; relu activation
        tf.keras.layers.Dense(units=8, activation='relu', input_shape=(X_train.shape[1],)),
        ## hidden 2: 4 nodes; relu activation
        tf.keras.layers.Dense(units=4, activation='relu'),
        ## output layer: 'num_classes' nodes; softmax activation
        tf.keras.layers.Dense(units=num_classes, activation='softmax') # Use num_classes for the output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
                             metrics=['accuracy'])

    model.summary()

    print("\n--- Training Multi-Class Model ---")
    history_multiclass = model.fit(
        X_train, y_train,
        epochs=50,       # Number of times to iterate over the entire training dataset
        batch_size=10,   # Number of samples per gradient update
        validation_split=0.1 # Use 10% of training data for validation during training
    )

    # 5. Evaluate the Model
    print("\n--- Evaluating Multi-Class Model ---")
    loss_mc, accuracy_mc = model.evaluate(X_test, y_test, verbose=2) # verbose=2 prints one line per epoch
    print(f"Test Loss (Multi-Class): {loss_mc:.4f}")
    print(f"Test Accuracy (Multi-Class): {accuracy_mc:.4f}")