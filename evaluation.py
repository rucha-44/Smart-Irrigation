import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, r2_score, mean_squared_error, mean_absolute_error)
from utils.data_processor import IrrigationDataProcessor

# Config (Must match training config)
DATA_PATH = 'multi_crop_realistic_dataset.csv'
MODEL_DIR = 'models/saved/'
PLOTS_DIR = 'models/plots/'
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred):
    """Generates and saves the Classification Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Water', 'Needs Water'],
                yticklabels=['No Water', 'Needs Water'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Classifier Confusion Matrix')
    save_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f" 📊 Saved Confusion Matrix to {save_path}")

def plot_regression_results(y_true, y_pred, r2):
    """Generates Actual vs Predicted and Residual plots for Regression"""
    
    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', edgecolors='k', s=20)
    
    # Perfect fit line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    plt.xlabel('Actual Water Amount (Liters)')
    plt.ylabel('Predicted Water Amount (Liters)')
    plt.title(f'Regression: Actual vs Predicted (R² = {r2:.4f})') # Added R2 to title
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Residual Plot
    residuals = y_true - y_pred.flatten()
    
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.xlabel('Prediction Error (Residuals)')
    plt.title('Error Distribution (Residuals)')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(PLOTS_DIR, 'regression_performance.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" 📊 Saved Regression Plots to {save_path}")

def evaluate():
    print("\n" + "="*70)
    print(" 🔍 STARTING MODEL EVALUATION & VISUALIZATION ")
    print("="*70)

    # 1. Load and Process Data
    print("\n[PHASE 1] Loading Test Data...")
    processor = IrrigationDataProcessor(DATA_PATH)
    try:
        processor.load_data()
        processor.clean_data()
        processor.preprocess() 
        
        # Split again to get test set
        _, X_test, _, y_c_test, _, y_r_test = processor.split_data()
        
        print(f"   > Test Set Shape: {X_test.shape}")
    except Exception as e:
        print(f"❌ Data Error: {e}")
        sys.exit(1)

    # 2. Load Models
    print("\n[PHASE 2] Loading Trained Models...")
    try:
        clf_path = os.path.join(MODEL_DIR, 'dnn_classifier.h5')
        reg_path = os.path.join(MODEL_DIR, 'dnn_regressor.h5')
        
        if not os.path.exists(clf_path) or not os.path.exists(reg_path):
            raise FileNotFoundError("Model files not found. Run train_system.py first.")

        # FIX: compile=False prevents the 'metrics deserialization' error
        classifier = tf.keras.models.load_model(clf_path, compile=False)
        regressor = tf.keras.models.load_model(reg_path, compile=False)
        print("   > Models loaded successfully.")
    except Exception as e:
        print(f"❌ Model Error: {e}")
        sys.exit(1)

    # 3. Evaluate Classifier
    print("\n[PHASE 3] Evaluating Classification Model...")
    y_pred_prob = classifier.predict(X_test, verbose=0)
    y_pred_c = (y_pred_prob > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_c_test, y_pred_c)
    prec = precision_score(y_c_test, y_pred_c)
    rec = recall_score(y_c_test, y_pred_c)
    f1 = f1_score(y_c_test, y_pred_c)

    print(f"   > Accuracy:  {acc*100:.2f}%")
    print(f"   > Precision: {prec:.4f}")
    print(f"   > Recall:    {rec:.4f}")
    print(f"   > F1 Score:  {f1:.4f}")
    
    plot_confusion_matrix(y_c_test, y_pred_c)

    # 4. Evaluate Regressor
    print("\n[PHASE 4] Evaluating Regression Model...")
    y_pred_scaled = regressor.predict(X_test, verbose=0)
    
    # Inverse Transform (Convert back to Liters)
    y_pred_real = processor.target_scaler.inverse_transform(y_pred_scaled)
    y_test_real = processor.target_scaler.inverse_transform(y_r_test)

    # Metrics
    # --- R2 SCORE CALCULATION ---
    r2 = r2_score(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_real, y_pred_real)

    print(f"   > R² Score: {r2:.4f}")  # <--- Printed here
    print(f"   > RMSE:     {rmse:.4f} Liters")
    print(f"   > MAE:      {mae:.4f} Liters")

    # Pass R2 to plotting function to show in title
    plot_regression_results(y_test_real, y_pred_real, r2)

    print("\n" + "="*70)
    print(" ✅ EVALUATION COMPLETE. Check 'models/plots/' for images.")
    print("="*70 + "\n")

if __name__ == "__main__":
    evaluate()