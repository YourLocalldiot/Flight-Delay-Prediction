"""
Model Utility Functions for Serialization and Loading

This module provides helper functions to save and load machine learning models
using joblib for efficient serialization.

Serialization: Converting Python objects into bytes for storage.
Joblib: Fast and efficient library for saving/loading Python objects,
especially useful for sklearn models and numpy arrays.
"""

import joblib
from pathlib import Path
from typing import Any, Dict, Optional


def save_model(
    model: Any,
    feature_names: list,
    scaler: Optional[Any] = None,
    model_name: str = "model",
    model_dir: Path = Path("../models"),
) -> Path:
    """
    Save a trained model and optional scaler using joblib.

    Args:
        model: Trained sklearn model object
        feature_names: List of feature names used in training
        scaler: Optional StandardScaler or other preprocessor
        model_name: Name for the model file (without extension)
        model_dir: Directory to save the model

    Returns:
        Path: Path to saved model file

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> save_model(model, feature_names=['age', 'price'], model_name='lr_model')
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a dictionary with model and metadata
    model_package = {
        "model": model,
        "feature_names": feature_names,
        "scaler": scaler,
        "model_type": model.__class__.__name__,
    }

    # Save using joblib
    model_file = model_dir / f"{model_name}.joblib"
    joblib.dump(model_package, model_file, compress=3)

    print(f"✓ Model saved to {model_file}")
    print(f"  Model type: {model.__class__.__name__}")
    print(f"  Features: {len(feature_names)}")
    if scaler:
        print(f"  Scaler: {scaler.__class__.__name__}")

    return model_file


def load_model(model_path: Path) -> Dict[str, Any]:
    """
    Load a trained model and metadata from joblib file.

    Args:
        model_path: Path to the saved model file

    Returns:
        Dict containing:
            - model: Trained sklearn model
            - feature_names: List of feature names
            - scaler: Optional preprocessor
            - model_type: String name of model class

    Raises:
        FileNotFoundError: If model file does not exist

    Example:
        >>> loaded = load_model(Path('../models/lr_model.joblib'))
        >>> model = loaded['model']
        >>> predictions = model.predict(X_test)
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_package = joblib.load(model_path)

    print(f"✓ Model loaded from {model_path}")
    print(f"  Model type: {model_package['model_type']}")
    print(f"  Features: {len(model_package['feature_names'])}")

    return model_package


def predict_with_preprocessing(
    model_package: Dict[str, Any], input_data, feature_names: list
):
    """
    Make predictions with automatic preprocessing.

    Args:
        model_package: Dictionary from load_model()
        input_data: Input features (dict or array-like)
        feature_names: List of feature names in correct order

    Returns:
        Prediction value or array
    """
    model = model_package["model"]
    scaler = model_package["scaler"]

    # Convert dict to array if needed
    if isinstance(input_data, dict):
        input_array = [input_data[name] for name in feature_names]
    else:
        input_array = input_data

    # Reshape for single sample if needed
    if len(input_array) == len(feature_names):
        input_array = [input_array]

    # Apply scaler if available (for Linear Regression)
    if scaler is not None:
        input_array = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_array)

    return prediction[0] if len(prediction) == 1 else prediction


if __name__ == "__main__":
    print("Model utilities module loaded. Use save_model() and load_model() functions.")
