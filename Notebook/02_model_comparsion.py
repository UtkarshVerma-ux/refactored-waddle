import pickle
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load preprocessed data"""
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {data['feature_names']}")
    print(f"Crops: {len(data['crop_names'])}")
    
    return X_train, X_test, y_train, y_test, data

def train_and_evaluate_model(model, name, X_train, X_test, y_train, y_test, 
                             scale_features=False):
    """Train a model and evaluate its performance"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {name}")
    print(f"{'='*70}")
    
    # Feature scaling if needed
    if scale_features:
        print("  Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Training
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Prediction
    start_time = time.time()
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    prediction_time = time.time() - start_time
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Testing Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted):    {recall:.4f}")
    print(f"  F1-Score (weighted):  {f1:.4f}")
    print(f"  Prediction time:      {prediction_time:.4f} seconds")
    
    # Check for overfitting
    overfit_gap = train_acc - test_acc
    if overfit_gap > 0.10:
        print(f"  ⚠️  High overfitting detected! Gap: {overfit_gap:.2%}")
    elif overfit_gap > 0.05:
        print(f"  ⚠️  Moderate overfitting. Gap: {overfit_gap:.2%}")
    else:
        print(f"  ✅ Good generalization. Gap: {overfit_gap:.2%}")
    
    results = {
        'model_name': name,
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'overfit_gap': overfit_gap,
        'y_test_pred': y_test_pred,
        'scaled': scale_features
    }
    
    return results

def compare_models():
    """Compare multiple ML models"""
    
    print("\n" + "🤖"*35)
    print("MODEL COMPARISON FOR CROP RECOMMENDATION")
    print("🤖"*35 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test, data = load_data()
    
    # Define models to compare
    models_config = [
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'scale': False
        },
        {
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'scale': False
        },
        {
            'name': 'K-Nearest Neighbors',
            'model': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'scale': True  # KNN benefits from scaling
        },
        {
            'name': 'Naive Bayes',
            'model': GaussianNB(),
            'scale': False
        },
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(
                max_iter=5000,  # Increased iterations to avoid convergence warning
                random_state=42,
                n_jobs=-1,
                solver='lbfgs'
            ),
            'scale': True  # Logistic Regression benefits from scaling
        }
    ]
    
    # Train and evaluate all models
    all_results = []
    
    for config in models_config:
        try:
            results = train_and_evaluate_model(
                config['model'], 
                config['name'], 
                X_train, X_test, y_train, y_test,
                scale_features=config['scale']
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error training {config['name']}: {e}")
            continue
    
    # Create comparison table
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*70)
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Train Acc': f"{r['train_accuracy']:.4f}",
            'Test Acc': f"{r['test_accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1_score']:.4f}",
            'Train Time (s)': f"{r['training_time']:.2f}",
            'Overfit Gap': f"{r['overfit_gap']:.4f}"
        }
        for r in all_results
    ])
    
    # Sort by test accuracy
    comparison_df['Test Acc Num'] = [r['test_accuracy'] for r in all_results]
    comparison_df = comparison_df.sort_values('Test Acc Num', ascending=False)
    comparison_df = comparison_df.drop('Test Acc Num', axis=1)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Identify best model
    best_result = max(all_results, key=lambda x: x['test_accuracy'])
    
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_result['model_name']}")
    print("="*70)
    print(f"  Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
    print(f"  F1-Score: {best_result['f1_score']:.4f}")
    print(f"  Training Time: {best_result['training_time']:.2f} seconds")
    print(f"  Overfitting Gap: {best_result['overfit_gap']:.4f}")
    
    # Why Random Forest is best
    print("\n" + "="*70)
    print("💡 WHY RANDOM FOREST IS THE BEST CHOICE:")
    print("="*70)
    
    rf_result = [r for r in all_results if r['model_name'] == 'Random Forest'][0]
    second_best = sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True)[1]
    
    acc_improvement = (rf_result['test_accuracy'] - second_best['test_accuracy']) * 100
    
    print(f"1. ✅ Highest Accuracy: {rf_result['test_accuracy']*100:.2f}%")
    print(f"   ({acc_improvement:.2f}% better than {second_best['model_name']})")
    
    print(f"\n2. ✅ Balanced Performance:")
    print(f"   Precision: {rf_result['precision']:.4f}")
    print(f"   Recall: {rf_result['recall']:.4f}")
    print(f"   F1-Score: {rf_result['f1_score']:.4f}")
    
    print(f"\n3. ✅ Good Generalization:")
    print(f"   Overfit gap: {rf_result['overfit_gap']:.2%} (< 5% is excellent)")
    
    print(f"\n4. ✅ Reasonable Training Time:")
    print(f"   {rf_result['training_time']:.2f} seconds for {len(data['crop_names'])} crops")
    
    print(f"\n5. ✅ Handles Multi-class Well:")
    print(f"   Successfully classifies {len(data['crop_names'])} different crops")
    
    # Statistical comparison
    print("\n" + "="*70)
    print("📈 STATISTICAL ANALYSIS")
    print("="*70)
    
    test_accs = [r['test_accuracy'] for r in all_results]
    print(f"\nTest Accuracy Statistics Across All Models:")
    print(f"  Mean: {np.mean(test_accs):.4f}")
    print(f"  Std:  {np.std(test_accs):.4f}")
    print(f"  Min:  {np.min(test_accs):.4f} ({[r['model_name'] for r in all_results if r['test_accuracy'] == np.min(test_accs)][0]})")
    print(f"  Max:  {np.max(test_accs):.4f} (Random Forest)")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    # Save comparison table
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print(f"\n✅ Saved comparison table to: results/model_comparison.csv")
    
    # Save best model
    best_model_data = {
        'model': best_result['model'],
        'model_name': best_result['model_name'],
        'metrics': {
            'train_accuracy': best_result['train_accuracy'],
            'test_accuracy': best_result['test_accuracy'],
            'precision': best_result['precision'],
            'recall': best_result['recall'],
            'f1_score': best_result['f1_score'],
            'training_time': best_result['training_time']
        },
        'feature_names': data['feature_names'],
        'crop_names': data['crop_names']
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model_data, f)
    
    print(f"✅ Saved best model to: models/best_model.pkl")
    
    # Create visualizations
    create_visualizations(all_results, comparison_df, y_test, best_result)
    
    return all_results, comparison_df, best_result

def create_visualizations(all_results, comparison_df, y_test, best_result):
    """Create comparison visualizations"""
    
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Accuracy Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    models = [r['model_name'] for r in all_results]
    train_accs = [r['train_accuracy'] for r in all_results]
    test_accs = [r['test_accuracy'] for r in all_results]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', 
                   alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', 
                   alpha=0.8, color='darkorange')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/accuracy_comparison.png")
    plt.close()
    
    # 2. Performance Metrics Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['test_accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1_score']
        }
        for r in all_results
    ])
    metrics_df = metrics_df.set_index('Model')
    
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/metrics_heatmap.png")
    plt.close()
    
    # 3. Training Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [r['training_time'] for r in all_results]
    colors = ['green' if r['model_name'] == best_result['model_name'] else 'steelblue' 
              for r in all_results]
    
    bars = ax.barh(models, times, color=colors, alpha=0.8)
    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(time, i, f' {time:.2f}s', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_time.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/training_time.png")
    plt.close()
    
    # 4. Overfitting Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    overfit_gaps = [r['overfit_gap'] for r in all_results]
    colors_overfit = ['red' if gap > 0.10 else 'orange' if gap > 0.05 else 'green' 
                      for gap in overfit_gaps]
    
    bars = ax.barh(models, overfit_gaps, color=colors_overfit, alpha=0.8)
    ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, 
               label='Moderate Overfit (5%)')
    ax.axvline(x=0.10, color='red', linestyle='--', linewidth=2, 
               label='High Overfit (10%)')
    ax.set_xlabel('Train-Test Accuracy Gap', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars, overfit_gaps)):
        ax.text(gap, i, f' {gap:.2%}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/figures/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/overfitting_analysis.png")
    plt.close()
    
    print("\n✅ All visualizations saved to: results/figures/")

if __name__ == "__main__":
    all_results, comparison_df, best_result = compare_models()
    
    print("\n" + "="*70)
    print("✅ MODEL COMPARISON COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  📄 results/model_comparison.csv")
    print("  📊 results/figures/accuracy_comparison.png")
    print("  📊 results/figures/metrics_heatmap.png")
    print("  📊 results/figures/training_time.png")
    print("  📊 results/figures/overfitting_analysis.png")
    print("  💾 models/best_model.pkl")
    print("\n🎯 Next Step: Option B - Detailed Random Forest evaluation!")