import pickle
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, classification_report, confusion_matrix)
import os

# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_features(X):
    """
    Apply feature engineering to create meaningful agricultural features
    
    Input: DataFrame with 7 original features
    Output: DataFrame with 14 features (7 original + 7 engineered)
    
    Engineered features:
    1. N_P_ratio: Nitrogen-Phosphorus balance
    2. N_K_ratio: Nitrogen-Potassium balance  
    3. P_K_ratio: Phosphorus-Potassium balance
    4. NPK_sum: Total soil fertility
    5. temp_humidity: Heat stress index
    6. ph_acidic: Binary flag for acidic soil (pH < 6.5)
    7. ph_alkaline: Binary flag for alkaline soil (pH > 7.5)
    """
    X_eng = X.copy()
    
    # NPK ratio features (nutrient balance indicators)
    X_eng['N_P_ratio'] = X['N'] / (X['P'] + 1)  # +1 to avoid division by zero
    X_eng['N_K_ratio'] = X['N'] / (X['K'] + 1)
    X_eng['P_K_ratio'] = X['P'] / (X['K'] + 1)
    
    # Total nutrient availability
    X_eng['NPK_sum'] = X['N'] + X['P'] + X['K']
    
    # Climate interaction (heat stress indicator)
    X_eng['temp_humidity'] = X['temperature'] * X['humidity']
    
    # pH category flags (important for crop selection)
    X_eng['ph_acidic'] = (X['ph'] < 6.5).astype(int)
    X_eng['ph_alkaline'] = (X['ph'] > 7.5).astype(int)
    
    return X_eng

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_complete_model():
    """
    Complete training pipeline with all optimizations
    Returns: Dictionary with model, performance metrics, and metadata
    """
    
    print("\n" + "="*70)
    print("RANDOM FOREST TRAINING PIPELINE FOR XAI FRAMEWORK")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n📂 STEP 1: Loading Data")
    print("-"*70)
    
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train_original = data['X_train']
    X_test_original = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    original_features = data['feature_names']
    crop_names = data['crop_names']
    
    print(f"✅ Data loaded successfully")
    print(f"   Training samples: {len(X_train_original)}")
    print(f"   Testing samples: {len(X_test_original)}")
    print(f"   Number of crops: {len(crop_names)}")
    print(f"   Original features: {original_features}")
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    
    print("\n⚙️  STEP 2: Applying Feature Engineering")
    print("-"*70)
    
    X_train = engineer_features(X_train_original)
    X_test = engineer_features(X_test_original)
    
    engineered_features = list(X_train.columns)
    new_features = [f for f in engineered_features if f not in original_features]
    
    print(f"✅ Feature engineering applied")
    print(f"   Original features: {len(original_features)}")
    print(f"   Engineered features: {len(engineered_features)}")
    print(f"   New features added: {new_features}")
    
    # ========================================================================
    # STEP 3: TRAIN MODEL
    # ========================================================================
    
    print("\n🌲 STEP 3: Training Optimized Random Forest")
    print("-"*70)
    
    # Optimized hyperparameters (from tuning)
    model_params = {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"   Hyperparameters:")
    for param, value in model_params.items():
        if param != 'n_jobs':
            print(f"     {param}: {value}")
    
    model = RandomForestClassifier(**model_params)
    
    print(f"\n   Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    
    # ========================================================================
    # STEP 4: EVALUATE PERFORMANCE
    # ========================================================================
    
    print("\n📊 STEP 4: Evaluating Model Performance")
    print("-"*70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\n   📈 Accuracy:")
    print(f"      Training:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"      Testing:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"      Gap:       {(train_acc - test_acc):.4f} ({(train_acc - test_acc)*100:.2f}%)")
    
    print(f"\n   📈 Other Metrics:")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    
    # Cross-validation
    print(f"\n   🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1)
    
    print(f"      Fold 1: {cv_scores[0]:.4f}")
    print(f"      Fold 2: {cv_scores[1]:.4f}")
    print(f"      Fold 3: {cv_scores[2]:.4f}")
    print(f"      Fold 4: {cv_scores[3]:.4f}")
    print(f"      Fold 5: {cv_scores[4]:.4f}")
    print(f"      Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    if cv_scores.std() < 0.02:
        print(f"      ✅ Excellent consistency (std < 2%)")
    elif cv_scores.std() < 0.05:
        print(f"      ✅ Good consistency (std < 5%)")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': engineered_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n   🌟 Top 10 Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"      {row['Feature']:20} {row['Importance']:.4f}")
    
    # ========================================================================
    # STEP 5: SAVE MODEL
    # ========================================================================
    
    print("\n💾 STEP 5: Saving Model for XAI Framework")
    print("-"*70)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Prepare complete model package
    model_package = {
        # Model
        'model': model,
        
        # Feature engineering
        'feature_engineering_applied': True,
        'engineer_features_function': engineer_features,
        'original_features': original_features,
        'engineered_features': engineered_features,
        'new_features': new_features,
        
        # Data info
        'crop_names': crop_names,
        'n_crops': len(crop_names),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        
        # Hyperparameters
        'hyperparameters': model_params,
        
        # Performance metrics
        'performance': {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'training_time': training_time
        },
        
        # Feature importance
        'feature_importance': importance_df.to_dict(),
        
        # For XAI framework
        'ready_for_xai': True,
        'xai_notes': 'Use engineer_features_function before prediction!'
    }
    
    # Save model
    with open('models/RF_MODEL_FOR_XAI.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"   ✅ Model saved: models/RF_MODEL_FOR_XAI.pkl")
    
    # Save feature importance
    importance_df.to_csv('results/feature_importance.csv', index=False)
    print(f"   ✅ Feature importance saved: results/feature_importance.csv")
    
    # Save performance summary
    performance_summary = pd.DataFrame([{
        'Model': 'Random Forest (Optimized + Feature Engineering)',
        'Train Accuracy': f"{train_acc:.4f}",
        'Test Accuracy': f"{test_acc:.4f}",
        'F1-Score': f"{f1:.4f}",
        'CV Mean': f"{cv_scores.mean():.4f}",
        'CV Std': f"{cv_scores.std():.4f}",
        'Training Time (s)': f"{training_time:.2f}"
    }])
    
    performance_summary.to_csv('results/model_performance_summary.csv', index=False)
    print(f"   ✅ Performance summary saved: results/model_performance_summary.csv")
    
    # Create usage guide for XAI implementation
    usage_guide = f"""
# HOW TO USE THIS MODEL IN XAI FRAMEWORK

## Load Model:
```python
import pickle

with open('models/RF_MODEL_FOR_XAI.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
engineer_features = model_package['engineer_features_function']
crop_names = model_package['crop_names']
```

## Make Predictions (IMPORTANT - Apply Feature Engineering!):
```python
import pandas as pd

# Your input
sample = {{
    'N': 78, 'P': 42, 'K': 43,
    'temperature': 25, 'humidity': 80,
    'ph': 6.5, 'rainfall': 1200
}}

X = pd.DataFrame([sample])

# CRITICAL: Apply feature engineering!
X_engineered = engineer_features(X)  # 7 → 14 features

# Predict
prediction = model.predict(X_engineered)[0]
probabilities = model.predict_proba(X_engineered)[0]

print(f"Predicted crop: {{prediction}}")
```

## Model Info:
- Test Accuracy: {test_acc*100:.2f}%
- F1-Score: {f1:.4f}
- Cross-Validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%
- Features: {len(engineered_features)} ({len(original_features)} original + {len(new_features)} engineered)
- Crops: {len(crop_names)}

## Engineered Features:
{', '.join(new_features)}

## IMPORTANT FOR XAI LAYERS:
1. SHAP: Use X_engineered (14 features)
2. LIME: Use X_engineered (14 features)
3. DiCE: Use X_engineered, but only vary controllable features!
"""
    
    with open('models/XAI_USAGE_GUIDE.txt', 'w', encoding='utf-8') as f:
        f.write(usage_guide)
    
    print(f"   ✅ Usage guide saved: models/XAI_USAGE_GUIDE.txt")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n🎯 Final Model Performance:")
    print(f"   Test Accuracy:  {test_acc*100:.2f}%")
    print(f"   F1-Score:       {f1:.4f}")
    print(f"   CV:             {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    print(f"\n📦 Model Package Contents:")
    print(f"   ✓ Trained Random Forest model")
    print(f"   ✓ Feature engineering function")
    print(f"   ✓ Hyperparameters")
    print(f"   ✓ Performance metrics")
    print(f"   ✓ Feature importance")
    print(f"   ✓ Crop names list")
    
    print(f"\n📁 Files Created:")
    print(f"   • models/RF_MODEL_FOR_XAI.pkl")
    print(f"   • models/XAI_USAGE_GUIDE.txt")
    print(f"   • results/feature_importance.csv")
    print(f"   • results/model_performance_summary.csv")
    return model_package

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🌾"*35)
    print("COMPLETE RANDOM FOREST TRAINING")
    print("🌾"*35)
    
    model_package = train_complete_model()
    
    print("\n✅ All done! Model ready for XAI implementation.\n")