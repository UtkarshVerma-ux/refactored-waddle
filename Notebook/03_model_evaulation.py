import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_data_and_model():
    """Load data and train Random Forest"""
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    crop_names = data['crop_names']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    print(f"Crops: {len(crop_names)}")
    
    # Train Random Forest
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST MODEL")
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
    print("✅ Model trained successfully!")
    
    return model, X_train, X_test, y_train, y_test, feature_names, crop_names

def evaluate_overall_performance(model, X_train, X_test, y_train, y_test):
    """Evaluate overall model performance"""
    
    print("\n" + "="*70)
    print("1. OVERALL PERFORMANCE METRICS")
    print("="*70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 Accuracy:")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Testing Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Overfitting Gap:     {(train_acc - test_acc):.4f} ({(train_acc - test_acc)*100:.2f}%)")
    
    print(f"\n📊 Precision:")
    print(f"  Macro Average:       {precision_macro:.4f}")
    print(f"  Weighted Average:    {precision_weighted:.4f}")
    
    print(f"\n📊 Recall:")
    print(f"  Macro Average:       {recall_macro:.4f}")
    print(f"  Weighted Average:    {recall_weighted:.4f}")
    
    print(f"\n📊 F1-Score:")
    print(f"  Macro Average:       {f1_macro:.4f}")
    print(f"  Weighted Average:    {f1_weighted:.4f}")
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'y_test_pred': y_test_pred
    }
    
    return metrics

def analyze_confusion_matrix(y_test, y_test_pred, crop_names):
    """Analyze confusion matrix"""
    
    print("\n" + "="*70)
    print("2. CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=crop_names)
    
    # Overall statistics
    total_predictions = cm.sum()
    correct_predictions = np.diag(cm).sum()
    incorrect_predictions = total_predictions - correct_predictions
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Predictions:     {total_predictions}")
    print(f"  Correct Predictions:   {correct_predictions} ({correct_predictions/total_predictions*100:.2f}%)")
    print(f"  Incorrect Predictions: {incorrect_predictions} ({incorrect_predictions/total_predictions*100:.2f}%)")
    
    # Most confused pairs
    print(f"\n❌ Top 10 Most Confused Crop Pairs:")
    print("-"*70)
    
    confusion_pairs = []
    for i in range(len(crop_names)):
        for j in range(len(crop_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'True': crop_names[i],
                    'Predicted': crop_names[j],
                    'Count': cm[i, j]
                })
    
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x['Count'], reverse=True)
    
    if confusion_pairs:
        for idx, pair in enumerate(confusion_pairs[:10], 1):
            print(f"{idx:2}. {pair['True']:20} → {pair['Predicted']:20} ({pair['Count']:2} times)")
    else:
        print("  ✅ No confusion! Perfect classification!")
    
    # Create confusion matrix visualization
    create_confusion_matrix_plot(cm, crop_names)
    
    return cm

def analyze_per_crop_performance(y_test, y_test_pred, crop_names):
    """Analyze performance for each crop"""
    
    print("\n" + "="*70)
    print("3. PER-CROP PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Generate classification report
    report = classification_report(y_test, y_test_pred, 
                                   labels=crop_names,
                                   output_dict=True,
                                   zero_division=0)
    
    # Convert to DataFrame
    crop_performance = []
    for crop in crop_names:
        if crop in report:
            crop_performance.append({
                'Crop': crop,
                'Precision': report[crop]['precision'],
                'Recall': report[crop]['recall'],
                'F1-Score': report[crop]['f1-score'],
                'Support': int(report[crop]['support'])
            })
    
    perf_df = pd.DataFrame(crop_performance)
    
    # Sort by F1-Score
    perf_df = perf_df.sort_values('F1-Score', ascending=False)
    
    print(f"\n📊 Crop Performance (sorted by F1-Score):")
    print("-"*70)
    print(perf_df.to_string(index=False))
    
    # Identify best and worst performers
    print(f"\n🏆 TOP 5 BEST PERFORMING CROPS:")
    print("-"*70)
    top_5 = perf_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"  {row['Crop']:20} - F1: {row['F1-Score']:.4f}, "
              f"Precision: {row['Precision']:.4f}, Recall: {row['Recall']:.4f}")
    
    print(f"\n⚠️  BOTTOM 5 CROPS NEEDING IMPROVEMENT:")
    print("-"*70)
    bottom_5 = perf_df.tail(5)
    for idx, row in bottom_5.iterrows():
        print(f"  {row['Crop']:20} - F1: {row['F1-Score']:.4f}, "
              f"Precision: {row['Precision']:.4f}, Recall: {row['Recall']:.4f}")
    
    # Save to CSV
    import os
    os.makedirs('results', exist_ok=True)
    perf_df.to_csv('results/per_crop_performance.csv', index=False)
    print(f"\n✅ Saved: results/per_crop_performance.csv")
    
    return perf_df

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance"""
    
    print("\n" + "="*70)
    print("4. FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Percentage': importances * 100
    }).sort_values('Importance', ascending=False)
    
    print(f"\n📊 Feature Importance Rankings:")
    print("-"*70)
    for idx, row in importance_df.iterrows():
        bar_length = int(row['Percentage'] / 2)  # Scale for display
        bar = '█' * bar_length
        print(f"{row['Feature']:15} {bar:30} {row['Percentage']:5.2f}%")
    
    print(f"\n💡 Interpretation:")
    print(f"  Most important: {importance_df.iloc[0]['Feature']} "
          f"({importance_df.iloc[0]['Percentage']:.2f}%)")
    print(f"  Least important: {importance_df.iloc[-1]['Feature']} "
          f"({importance_df.iloc[-1]['Percentage']:.2f}%)")
    
    # Top 3 features contribute what percentage?
    top3_contribution = importance_df.head(3)['Percentage'].sum()
    print(f"  Top 3 features contribute: {top3_contribution:.2f}% of total importance")
    
    # Create feature importance plot
    create_feature_importance_plot(importance_df)
    
    return importance_df

def perform_cross_validation(model, X_train, y_train):
    """Perform cross-validation"""
    
    print("\n" + "="*70)
    print("5. CROSS-VALIDATION ANALYSIS")
    print("="*70)
    
    print("\nPerforming 5-fold cross-validation...")
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1)
    
    print(f"\n📊 Cross-Validation Results:")
    print(f"  Fold 1: {cv_scores[0]:.4f} ({cv_scores[0]*100:.2f}%)")
    print(f"  Fold 2: {cv_scores[1]:.4f} ({cv_scores[1]*100:.2f}%)")
    print(f"  Fold 3: {cv_scores[2]:.4f} ({cv_scores[2]*100:.2f}%)")
    print(f"  Fold 4: {cv_scores[3]:.4f} ({cv_scores[3]*100:.2f}%)")
    print(f"  Fold 5: {cv_scores[4]:.4f} ({cv_scores[4]*100:.2f}%)")
    
    print(f"\n📈 Summary Statistics:")
    print(f"  Mean Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"  Std Deviation: {cv_scores.std():.4f} ({cv_scores.std()*100:.2f}%)")
    print(f"  Min Accuracy:  {cv_scores.min():.4f} ({cv_scores.min()*100:.2f}%)")
    print(f"  Max Accuracy:  {cv_scores.max():.4f} ({cv_scores.max()*100:.2f}%)")
    
    # Consistency check
    if cv_scores.std() < 0.02:
        print(f"\n  ✅ Excellent consistency across folds (std < 2%)")
    elif cv_scores.std() < 0.05:
        print(f"\n  ✅ Good consistency across folds (std < 5%)")
    else:
        print(f"\n  ⚠️  Moderate variance across folds (std > 5%)")
    
    return cv_scores

def analyze_errors(y_test, y_test_pred, X_test, crop_names, feature_names):
    """Analyze misclassified samples"""
    
    print("\n" + "="*70)
    print("6. ERROR ANALYSIS")
    print("="*70)
    
    # Find misclassified samples
    misclassified_mask = y_test != y_test_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    total_errors = len(misclassified_indices)
    error_rate = total_errors / len(y_test) * 100
    
    print(f"\n📊 Error Statistics:")
    print(f"  Total Misclassifications: {total_errors}")
    print(f"  Error Rate: {error_rate:.2f}%")
    print(f"  Accuracy: {100 - error_rate:.2f}%")
    
    if total_errors > 0:
        # Analyze error patterns
        error_crops = y_test.iloc[misclassified_indices]
        error_predictions = y_test_pred[misclassified_indices]
        
        # Most commonly misclassified crops
        error_counter = Counter(error_crops)
        print(f"\n❌ Crops Most Frequently Misclassified:")
        print("-"*70)
        for crop, count in error_counter.most_common(10):
            total_crop_samples = (y_test == crop).sum()
            error_pct = count / total_crop_samples * 100
            print(f"  {crop:20} {count:3} errors out of {total_crop_samples:3} samples ({error_pct:.1f}%)")
        
        # Show a few example errors
        print(f"\n📝 Sample Misclassifications (first 5):")
        print("-"*70)
        for i, idx in enumerate(misclassified_indices[:5]):
            true_crop = y_test.iloc[idx]
            pred_crop = y_test_pred[idx]
            print(f"\nError {i+1}:")
            print(f"  True Crop:      {true_crop}")
            print(f"  Predicted Crop: {pred_crop}")
            print(f"  Feature values:")
            for fname, fval in zip(feature_names, X_test.iloc[idx]):
                print(f"    {fname:15} = {fval}")
    else:
        print("\n✅ Perfect classification - No errors!")
    
    return total_errors, error_rate

def create_confusion_matrix_plot(cm, crop_names):
    """Create confusion matrix heatmap"""
    
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # For large number of crops, create a simplified version
    plt.figure(figsize=(16, 14))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, 
                xticklabels=crop_names, 
                yticklabels=crop_names,
                cmap='Blues', 
                cbar_kws={'label': 'Normalized Frequency'},
                linewidths=0.5,
                linecolor='gray',
                square=True)
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Crop', fontsize=12, fontweight='bold')
    plt.ylabel('True Crop', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/figures/confusion_matrix.png")
    plt.close()

def create_feature_importance_plot(importance_df):
    """Create feature importance visualization"""
    
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    
    plt.barh(importance_df['Feature'], importance_df['Percentage'], 
             color=colors, alpha=0.8)
    plt.xlabel('Importance (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(row['Percentage'], i, f" {row['Percentage']:.2f}%", 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: results/figures/feature_importance.png")
    plt.close()

def create_performance_summary_plot(perf_df):
    """Create per-crop performance visualization"""
    
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    # Sort by F1-Score for better visualization
    perf_df_sorted = perf_df.sort_values('F1-Score', ascending=True)
    
    y_pos = np.arange(len(perf_df_sorted))
    
    plt.barh(y_pos, perf_df_sorted['F1-Score'], alpha=0.8, color='steelblue', label='F1-Score')
    plt.barh(y_pos, perf_df_sorted['Precision'], alpha=0.6, color='orange', label='Precision')
    plt.barh(y_pos, perf_df_sorted['Recall'], alpha=0.6, color='green', label='Recall')
    
    plt.yticks(y_pos, perf_df_sorted['Crop'])
    plt.xlabel('Score', fontsize=12, fontweight='bold')
    plt.ylabel('Crop', fontsize=12, fontweight='bold')
    plt.title('Per-Crop Performance Metrics', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='x', alpha=0.3)
    plt.xlim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('results/figures/per_crop_performance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: results/figures/per_crop_performance.png")
    plt.close()

def main():
    """Main evaluation pipeline"""
    
    print("\n" + "🔬"*35)
    print("DETAILED RANDOM FOREST MODEL EVALUATION")
    print("🔬"*35 + "\n")
    
    # Load data and model
    model, X_train, X_test, y_train, y_test, feature_names, crop_names = load_data_and_model()
    
    # 1. Overall Performance
    metrics = evaluate_overall_performance(model, X_train, X_test, y_train, y_test)
    
    # 2. Confusion Matrix
    cm = analyze_confusion_matrix(y_test, metrics['y_test_pred'], crop_names)
    
    # 3. Per-Crop Performance
    perf_df = analyze_per_crop_performance(y_test, metrics['y_test_pred'], crop_names)
    create_performance_summary_plot(perf_df)
    
    # 4. Feature Importance
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 5. Cross-Validation
    cv_scores = perform_cross_validation(model, X_train, y_train)
    
    # 6. Error Analysis
    total_errors, error_rate = analyze_errors(y_test, metrics['y_test_pred'], 
                                              X_test, crop_names, feature_names)
    
    # Save comprehensive summary
    import os
    os.makedirs('results', exist_ok=True)
    
    summary = {
        'Model': 'Random Forest',
        'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
        'F1-Score (weighted)': f"{metrics['f1_weighted']:.4f}",
        'CV Mean Accuracy': f"{cv_scores.mean():.4f}",
        'CV Std': f"{cv_scores.std():.4f}",
        'Total Errors': total_errors,
        'Error Rate': f"{error_rate:.2f}%"
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('results/model_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  📄 results/model_summary.csv")
    print("  📄 results/per_crop_performance.csv")
    print("  📊 results/figures/confusion_matrix.png")
    print("  📊 results/figures/feature_importance.png")
    print("  📊 results/figures/per_crop_performance.png")
    
    print("\n🎯 Key Findings:")
    print(f"  • Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"  • Cross-Validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  • Most Important Feature: {importance_df.iloc[0]['Feature']}")
    print(f"  • Best Performing Crop: {perf_df.iloc[0]['Crop']} (F1: {perf_df.iloc[0]['F1-Score']:.4f})")
    
    print("\n🚀 Ready for next step: XAI Framework Implementation!")

if __name__ == "__main__":
    main()