"""
Comparison: Single Model vs pH-Specific Models
Tests if soil-condition-specific modeling improves performance

Approaches:
A. Single Random Forest (current)
B. Three pH-specific Random Forests (your idea)

Author: Research Framework  
Date: Feb 26, 2026
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

def load_data():
    """Load data"""
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def approach_a_single_model(X_train, X_test, y_train, y_test):
    """Approach A: Single Random Forest for all conditions"""
    
    print("\n" + "="*70)
    print("APPROACH A: SINGLE RANDOM FOREST MODEL")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Training samples: {len(X_train)}")
    
    return {'accuracy': accuracy, 'f1': f1, 'model': model, 'predictions': y_pred}

def approach_b_ph_specific_models(X_train, X_test, y_train, y_test):
    """Approach B: Three pH-specific Random Forest models"""
    
    print("\n" + "="*70)
    print("APPROACH B: pH-SPECIFIC RANDOM FOREST MODELS")
    print("="*70)
    
    # Define pH ranges
    ph_ranges = {
        'acidic': (0, 6.5),      # pH < 6.5
        'neutral': (6.5, 7.5),   # 6.5 ≤ pH ≤ 7.5
        'alkaline': (7.5, 14)    # pH > 7.5
    }
    
    # Split training data by pH
    train_splits = {}
    for condition, (ph_min, ph_max) in ph_ranges.items():
        mask = (X_train['ph'] > ph_min) & (X_train['ph'] <= ph_max)
        train_splits[condition] = {
            'X': X_train[mask],
            'y': y_train[mask]
        }
        print(f"\n{condition.upper()} soil training samples: {len(train_splits[condition]['X'])}")
    
    # Train separate models
    models = {}
    for condition in ph_ranges.keys():
        X_cond = train_splits[condition]['X']
        y_cond = train_splits[condition]['y']
        
        if len(X_cond) > 0:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_cond, y_cond)
            models[condition] = model
            print(f"  ✅ Trained {condition} model")
        else:
            models[condition] = None
            print(f"  ⚠️  No samples for {condition} soil!")
    
    # Predict on test set using appropriate model based on pH
    y_pred = []
    test_distribution = {'acidic': 0, 'neutral': 0, 'alkaline': 0}
    
    for idx in range(len(X_test)):
        sample = X_test.iloc[idx:idx+1]
        ph_value = sample['ph'].values[0]
        
        # Determine which model to use
        if ph_value <= 6.5:
            condition = 'acidic'
        elif ph_value <= 7.5:
            condition = 'neutral'
        else:
            condition = 'alkaline'
        
        test_distribution[condition] += 1
        
        # Predict using appropriate model
        if models[condition] is not None:
            pred = models[condition].predict(sample)[0]
        else:
            # Fallback to neutral model if specific model unavailable
            pred = models['neutral'].predict(sample)[0]
        
        y_pred.append(pred)
    
    print(f"\n📊 Test set distribution:")
    for condition, count in test_distribution.items():
        print(f"  {condition}: {count} samples ({count/len(X_test)*100:.1f}%)")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy, 
        'f1': f1, 
        'models': models, 
        'predictions': y_pred,
        'test_distribution': test_distribution
    }

def detailed_comparison(results_a, results_b, y_test):
    """Compare both approaches in detail"""
    
    print("\n" + "="*70)
    print("📊 DETAILED COMPARISON")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'F1-Score', 'Model Complexity'],
        'Single Model': [
            f"{results_a['accuracy']:.4f}",
            f"{results_a['f1']:.4f}",
            '1 model'
        ],
        'pH-Specific Models': [
            f"{results_b['accuracy']:.4f}",
            f"{results_b['f1']:.4f}",
            '3 models'
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Improvement calculation
    acc_improvement = (results_b['accuracy'] - results_a['accuracy']) * 100
    f1_improvement = (results_b['f1'] - results_a['f1']) * 100
    
    print(f"\n💡 Analysis:")
    if acc_improvement > 1:
        print(f"  ✅ pH-Specific models improve accuracy by {acc_improvement:.2f}%")
        print(f"     → Significant improvement! Use pH-specific approach.")
    elif acc_improvement > 0.5:
        print(f"  ⚠️  pH-Specific models improve accuracy by {acc_improvement:.2f}%")
        print(f"     → Marginal improvement. Consider trade-off with complexity.")
    elif acc_improvement > 0:
        print(f"  ⚠️  pH-Specific models improve accuracy by only {acc_improvement:.2f}%")
        print(f"     → Minimal improvement. Single model is simpler and sufficient.")
    else:
        print(f"  ❌ pH-Specific models REDUCE accuracy by {abs(acc_improvement):.2f}%")
        print(f"     → Single model is better! Stick with Approach A.")
    
    print(f"\n  F1-Score change: {f1_improvement:+.2f}%")
    
    # Agreement analysis
    predictions_match = np.array(results_a['predictions']) == np.array(results_b['predictions'])
    agreement_rate = predictions_match.sum() / len(predictions_match)
    
    print(f"\n🤝 Model Agreement:")
    print(f"  Both models agree on {agreement_rate*100:.2f}% of predictions")
    print(f"  Predictions differ on {(1-agreement_rate)*100:.2f}% of cases")
    
    # Where they differ, which is more accurate?
    if agreement_rate < 1.0:
        differ_indices = np.where(~predictions_match)[0]
        y_test_array = np.array(y_test)
        
        single_correct = (np.array(results_a['predictions'])[differ_indices] == 
                         y_test_array[differ_indices]).sum()
        ph_specific_correct = (np.array(results_b['predictions'])[differ_indices] == 
                               y_test_array[differ_indices]).sum()
        
        print(f"\n  On disagreements:")
        print(f"    Single model correct: {single_correct}/{len(differ_indices)}")
        print(f"    pH-specific correct: {ph_specific_correct}/{len(differ_indices)}")
    
    return acc_improvement, f1_improvement

def create_comparison_visualization(results_a, results_b):
    """Create visual comparison"""
    
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['Single\nModel', 'pH-Specific\nModels']
    accuracies = [results_a['accuracy'], results_b['accuracy']]
    colors = ['steelblue', 'darkorange']
    
    axes[0].bar(models, accuracies, color=colors, alpha=0.8, width=0.6)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0.9, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        axes[0].text(i, acc + 0.005, f'{acc:.4f}', ha='center', fontweight='bold')
    
    # F1-Score comparison
    f1_scores = [results_a['f1'], results_b['f1']]
    
    axes[1].bar(models, f1_scores, color=colors, alpha=0.8, width=0.6)
    axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0.9, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (model, f1) in enumerate(zip(models, f1_scores)):
        axes[1].text(i, f1 + 0.005, f'{f1:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/model_approach_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/figures/model_approach_comparison.png")
    plt.close()

def main():
    """Main comparison"""
    
    print("\n" + "🔬"*35)
    print("SINGLE vs pH-SPECIFIC MODELS COMPARISON")
    print("🔬"*35)
    
    # Load data
    data = load_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"pH distribution in training set:")
    print(f"  Mean: {X_train['ph'].mean():.2f}")
    print(f"  Std: {X_train['ph'].std():.2f}")
    print(f"  Min: {X_train['ph'].min():.2f}")
    print(f"  Max: {X_train['ph'].max():.2f}")
    
    # Test both approaches
    results_a = approach_a_single_model(X_train, X_test, y_train, y_test)
    results_b = approach_b_ph_specific_models(X_train, X_test, y_train, y_test)
    
    # Compare
    acc_improvement, f1_improvement = detailed_comparison(results_a, results_b, y_test)
    
    # Visualize
    create_comparison_visualization(results_a, results_b)
    
    # Recommendation
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION")
    print("="*70)
    
    if acc_improvement > 1:
        print("\n✅ USE pH-SPECIFIC MODELS")
        print("   Reason: Significant accuracy improvement (>1%)")
        print("   Benefit: Better crop-specific pH modeling")
        print("   Cost: Slightly more complex (3 models vs 1)")
    elif acc_improvement > 0:
        print("\n⚠️  MARGINAL BENEFIT")
        print(f"   pH-specific improves by only {acc_improvement:.2f}%")
        print("   Options:")
        print("     A. Use single model (simpler, sufficient)")
        print("     B. Use pH-specific (research novelty, marginal gain)")
        print("\n   For research paper: pH-specific adds novelty!")
    else:
        print("\n❌ STICK WITH SINGLE MODEL")
        print("   Reason: pH-specific does not improve performance")
        print("   Single model is simpler and equally/more accurate")
    
    print("\n✅ Comparison complete!")
    
    return results_a, results_b, acc_improvement

if __name__ == "__main__":
    results_a, results_b, improvement = main()