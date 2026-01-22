"""
Solar Energy Prediction Model Training Script
Converts the Model.ipynb notebook to a standalone Python script.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_and_prepare_data():
    """Load and prepare the solar data with feature engineering."""
    # Load and prepare data
    df = pd.read_csv("solar data.csv")
    df.columns = ['date', 'from_time', 'to_time', 'power_mw', 'drop_col']
    df = df.drop(columns=['drop_col'])

    # Create datetime column and extract time-based features
    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['from_time'],
        format='%d.%m.%Y %H:%M',
        errors='coerce'
    )

    # Remove any rows that couldn't be parsed
    df = df.dropna(subset=['datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_month'] = df['datetime'].dt.day

    # Create lag features
    df['power_prev_1h'] = df['power_mw'].shift(4)   # 4 * 15min = 1 hour
    df['power_prev_2h'] = df['power_mw'].shift(8)   # 2 hours
    df['power_prev_4h'] = df['power_mw'].shift(16)  # 4 hours

    # Calculate rolling statistics
    df['power_mean_24h'] = df['power_mw'].rolling(window=96).mean()  # 96 * 15min = 24 hours
    df['power_std_24h'] = df['power_mw'].rolling(window=96).std()

    # Drop rows with NaN from feature engineering
    df = df.dropna()

    print(f"Dataset shape after feature engineering: {df.shape}")
    print(f"Feature columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    return df


def split_data(df):
    """Split data into train, validation, and test sets."""
    feature_cols = ['hour', 'day_of_week', 'month', 'day_of_month',
                    'power_prev_1h', 'power_prev_2h', 'power_prev_4h',
                    'power_mean_24h', 'power_std_24h']

    X = df[feature_cols]
    y = df['power_mw']

    # Split data into train, validation, and test sets (60%, 20%, 20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\nData Split Summary:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"\nTarget Variable Statistics:")
    print(f"  Mean: {y.mean():.2f} MW")
    print(f"  Std: {y.std():.2f} MW")
    print(f"  Min: {y.min():.2f} MW")
    print(f"  Max: {y.max():.2f} MW")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_random_forest_model(X_train, y_train, feature_cols):
    """Train the Random Forest model."""
    # ============================================================================
    # TRAIN RANDOM FOREST MODEL
    # ============================================================================

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    print("✓ Model trained successfully")

    # Feature importance
    print(f"\nTop 5 Feature Importance:")
    feature_importance = sorted(zip(feature_cols, rf_model.feature_importances_),
                                key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")

    return rf_model


def evaluate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def evaluate_model(rf_model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate the model on all sets."""
    # ============================================================================
    # EVALUATE RANDOM FOREST ON ALL SETS
    # ============================================================================

    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)

    train_metrics = evaluate_metrics(y_train, y_train_pred)
    val_metrics = evaluate_metrics(y_val, y_val_pred)
    test_metrics = evaluate_metrics(y_test, y_test_pred)

    print("\n" + "-"*70)
    print("TRAINING SET METRICS")
    print("-"*70)
    for metric, value in train_metrics.items():
        print(f"  {metric:6s}: {value:10.4f}")

    print("\n" + "-"*70)
    print("VALIDATION SET METRICS")
    print("-"*70)
    for metric, value in val_metrics.items():
        print(f"  {metric:6s}: {value:10.4f}")

    print("\n" + "-"*70)
    print("TEST SET METRICS (Final Model Performance)")
    print("-"*70)
    for metric, value in test_metrics.items():
        print(f"  {metric:6s}: {value:10.4f}")

    return test_metrics


def save_model(model, filename='best_solar_model.pkl'):
    """Save the trained model to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Best model saved as '{filename}'")


def main():
    """Main training pipeline."""
    print("="*70)
    print("SOLAR ENERGY PREDICTION MODEL TRAINING")
    print("="*70)

    # Load and prepare data
    df = load_and_prepare_data()

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = split_data(df)

    # Train model
    rf_model = train_random_forest_model(X_train, y_train, feature_cols)

    # Evaluate model
    test_metrics = evaluate_model(rf_model, X_train, X_val, X_test, y_train, y_val, y_test)

    # ============================================================================
    # MODEL SUMMARY
    # ============================================================================

    print(f"\nRandom Forest Regressor Configuration:")
    print(f"  Number of Trees: {rf_model.n_estimators}")
    print(f"  Max Depth: {rf_model.max_depth}")
    print(f"  Number of Features: {len(feature_cols)}")
    print(f"\nTest Set Performance:")
    print(f"  ✓ RMSE (Root Mean Squared Error): {test_metrics['RMSE']:.4f} MW")
    print(f"  ✓ MAE (Mean Absolute Error): {test_metrics['MAE']:.4f} MW")
    print(f"  ✓ R² Score: {test_metrics['R2']:.4f}")
    print(f"  ✓ MAPE (Mean Absolute Percentage Error): {test_metrics['MAPE']:.4f}%")

    # Save the model
    save_model(rf_model)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
