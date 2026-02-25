import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import time

def load_data():
    """Load data"""
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def current_best_model(X_train, X_test, y_train, y_test):
    """Current optimized model"""
    
    print("="*70)
    print("CURRENT BEST MODEL (BASELINE)")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Current Performance:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {'accuracy': acc, 'f1': f1, 'name': 'Current Best', 'model': model}

def improvement_1_feature_engineering(X_train, X_test, y_train, y_test):
    """Test feature engineering"""
    
    print("\n" + "="*70)
    print("IMPROVEMENT 1: FEATURE ENGINEERING")
    print("="*70)
    
    print("\n⚙️  Creating engineered features...")
    
    # Create new features
    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()
    
    # NPK ratio features
    X_train_eng['N_P_ratio'] = X_train['N'] / (X_train['P'] + 1)  # +1 to avoid division by zero
    X_train_eng['N_K_ratio'] = X_train['N'] / (X_train['K'] + 1)
    X_train_eng['P_K_ratio'] = X_train['P'] / (X_train['K'] + 1)
    
    X_test_eng['N_P_ratio'] = X_test['N'] / (X_test['P'] + 1)
    X_test_eng['N_K_ratio'] = X_test['N'] / (X_test['K'] + 1)
    X_test_eng['P_K_ratio'] = X_test['P'] / (X_test['K'] + 1)
    
    # NPK sum
    X_train_eng['NPK_sum'] = X_train['N'] + X_train['P'] + X_train['K']
    X_test_eng['NPK_sum'] = X_test['N'] + X_test['P'] + X_test['K']
    
    # Climate interaction
    X_train_eng['temp_humidity'] = X_train['temperature'] * X_train['humidity']
    X_test_eng['temp_humidity'] = X_test['temperature'] * X_test['humidity']
    
    # pH categories
    X_train_eng['ph_acidic'] = (X_train['ph'] < 6.5).astype(int)
    X_train_eng['ph_alkaline'] = (X_train['ph'] > 7.5).astype(int)
    
    X_test_eng['ph_acidic'] = (X_test['ph'] < 6.5).astype(int)
    X_test_eng['ph_alkaline'] = (X_test['ph'] > 7.5).astype(int)
    
    print(f"  Original features: {len(X_train.columns)}")
    print(f"  Engineered features: {len(X_train_eng.columns)}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_eng, y_train)
    y_pred = model.predict(X_test_eng)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Performance with Feature Engineering:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {'accuracy': acc, 'f1': f1, 'name': 'Feature Engineering'}

def improvement_2_class_weights(X_train, X_test, y_train, y_test):
    """Test class weight balancing"""
    
    print("\n" + "="*70)
    print("IMPROVEMENT 2: CLASS WEIGHT BALANCING")
    print("="*70)
    
    # Check class distribution
    class_counts = Counter(y_train)
    print(f"\n📊 Class distribution (training set):")
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    print(f"  Most samples: {sorted_classes[-1][0]} ({sorted_classes[-1][1]} samples)")
    print(f"  Least samples: {sorted_classes[0][0]} ({sorted_classes[0][1]} samples)")
    print(f"  Imbalance ratio: {sorted_classes[-1][1] / sorted_classes[0][1]:.2f}:1")
    
    # Train with class weights
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',  # ← New!
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Performance with Class Weights:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {'accuracy': acc, 'f1': f1, 'name': 'Class Weights'}

def improvement_3_ensemble_voting(X_train, X_test, y_train, y_test):
    """Test ensemble voting classifier"""
    
    print("\n" + "="*70)
    print("IMPROVEMENT 3: ENSEMBLE VOTING (RF + GB)")
    print("="*70)
    
    print("\n⚙️  Training multiple models for ensemble...")
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Voting Classifier
    print("  Training Random Forest...")
    start = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    
    print("  Training Gradient Boosting...")
    start = time.time()
    gb.fit(X_train, y_train)
    gb_time = time.time() - start
    
    print(f"  RF training time: {rf_time:.2f}s")
    print(f"  GB training time: {gb_time:.2f}s")
    
    # Soft voting (average probabilities)
    y_pred_rf_proba = rf.predict_proba(X_test)
    y_pred_gb_proba = gb.predict_proba(X_test)
    
    # Average probabilities
    y_pred_proba = (y_pred_rf_proba + y_pred_gb_proba) / 2
    y_pred = rf.classes_[np.argmax(y_pred_proba, axis=1)]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Performance with Ensemble Voting:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Total training time: {rf_time + gb_time:.2f}s")
    
    return {'accuracy': acc, 'f1': f1, 'name': 'Ensemble Voting'}

def improvement_4_more_trees(X_train, X_test, y_train, y_test):
    """Test significantly more trees"""
    
    print("\n" + "="*70)
    print("IMPROVEMENT 4: MORE TREES (500)")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=500,  # ← Increased from 200
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Performance with 500 Trees:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Training time: {train_time:.2f}s")
    
    return {'accuracy': acc, 'f1': f1, 'name': 'More Trees (500)'}

def improvement_5_bootstrap_samples(X_train, X_test, y_train, y_test):
    """Test different bootstrap settings"""
    
    print("\n" + "="*70)
    print("IMPROVEMENT 5: BOOTSTRAP SETTINGS")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        max_samples=0.8,  # Use 80% of samples for each tree
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Performance with Bootstrap Tuning:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {'accuracy': acc, 'f1': f1, 'name': 'Bootstrap Tuning'}

def main():
    """Test all improvements"""
    
    print("\n" + "🔬"*35)
    print("COMPREHENSIVE IMPROVEMENT ANALYSIS")
    print("🔬"*35)
    
    # Load data
    data = load_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")
    print(f"Crops: {len(data['crop_names'])}")
    
    # Test all improvements
    results = []
    
    # Baseline
    baseline = current_best_model(X_train, X_test, y_train, y_test)
    results.append(baseline)
    
    # Improvement 1: Feature Engineering
    try:
        result = improvement_1_feature_engineering(X_train, X_test, y_train, y_test)
        results.append(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Improvement 2: Class Weights
    try:
        result = improvement_2_class_weights(X_train, X_test, y_train, y_test)
        results.append(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Improvement 3: Ensemble
    try:
        result = improvement_3_ensemble_voting(X_train, X_test, y_train, y_test)
        results.append(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Improvement 4: More Trees
    try:
        result = improvement_4_more_trees(X_train, X_test, y_train, y_test)
        results.append(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Improvement 5: Bootstrap
    try:
        result = improvement_5_bootstrap_samples(X_train, X_test, y_train, y_test)
        results.append(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON")
    print("="*70)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    print("\n" + comparison_df[['name', 'accuracy', 'f1']].to_string(index=False))
    
    # Best improvement
    best = comparison_df.iloc[0]
    baseline_acc = baseline['accuracy']
    improvement = (best['accuracy'] - baseline_acc) * 100
    
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION")
    print("="*70)
    
    if best['name'] == 'Current Best':
        print("\n✅ CURRENT MODEL IS ALREADY OPTIMAL")
        print("   No improvements found from tested approaches")
        print(f"   Current accuracy: {baseline_acc*100:.2f}%")
    else:
        print(f"\n🏆 BEST APPROACH: {best['name']}")
        print(f"   Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
        print(f"   Improvement: {improvement:+.2f}%")
        
        if improvement > 0.5:
            print(f"\n✅ RECOMMENDED: Use {best['name']}")
        elif improvement > 0.2:
            print(f"\n⚠️  MARGINAL: {best['name']} improves by {improvement:.2f}%")
            print(f"   Consider trade-off with complexity")
        else:
            print(f"\n⚠️  MINIMAL: {best['name']} improves by only {improvement:.2f}%")
            print(f"   Current model is sufficient")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    comparison_df.to_csv('results/all_improvements_comparison.csv', index=False)
    print(f"\n✅ Saved: results/all_improvements_comparison.csv")
    
    return comparison_df

if __name__ == "__main__":
    results = main()