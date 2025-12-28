import os
import sys
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from utils.data_processor import IrrigationDataProcessor
from models.architectures import build_advanced_dnn


# Config
DATA_PATH = 'multi_crop_realistic_dataset.csv'
MODEL_DIR = 'models/saved/'
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print("\n" + "#"*70)
    print(" 🚀 STARTING STABLE TRAINING (Advanced DNN) ")
    print("#"*70)

    # 1. Data Processing
    print("\n[PHASE 1] Processing Data...")
    processor = IrrigationDataProcessor(DATA_PATH)
    try:
        processor.load_data()
        processor.clean_data()
        processor.preprocess()
        X_train, X_test, y_c_train, y_c_test, y_r_train, y_r_test = processor.split_data()
        
        # Save AFTER split (Fixes "Not Fitted" error)
        processor.save_processors(MODEL_DIR)
        print(f"   > Train Shape: {X_train.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-6, verbose=1),
        TerminateOnNaN()
    ]

    # 2. Train Classifier
    print("\n[PHASE 2] Training Classifier...")
    clf = build_advanced_dnn(X_train.shape[1], 'classification')
    clf.fit(X_train, y_c_train, validation_split=0.2, epochs=80, batch_size=64, verbose=1, callbacks=callbacks)
    clf.save(os.path.join(MODEL_DIR, 'dnn_classifier.h5'))
    
    y_pred_c = (clf.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_c_test, y_pred_c)
    print(f"   > Classifier Accuracy: {acc*100:.2f}%")

    # 3. Train Regressor
    print("\n[PHASE 3] Training Regressor...")
    reg = build_advanced_dnn(X_train.shape[1], 'regression')
    reg.fit(X_train, y_r_train, validation_split=0.2, epochs=120, batch_size=64, verbose=0, callbacks=callbacks)
    reg.save(os.path.join(MODEL_DIR, 'dnn_regressor.h5'))
    
    # Eval
    y_pred_scaled = reg.predict(X_test)
    y_pred_real = processor.target_scaler.inverse_transform(y_pred_scaled)
    y_test_real = processor.target_scaler.inverse_transform(y_r_test)
    
    r2 = r2_score(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)

    print("\n" + "="*60)
    print(f"{'🏆 FINAL RESULTS':^60}")
    print("="*60)
    print(f"  ✅ Classifier Accuracy: {acc*100:.2f}%")
    print(f"  ✅ Regressor R² Score:  {r2:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    train()