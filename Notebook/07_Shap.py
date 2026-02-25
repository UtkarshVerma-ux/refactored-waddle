"""
LAYER 1: SHAP - ENHANCED GLOBAL EXPLAINABILITY
Enhanced version with:
- Dependence plots (feature interactions)
- Per-crop SHAP analysis
- SHAP Explanation objects

Author: Research Framework
Date: Feb 26, 2026
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import os

# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_features(X):
    """Apply feature engineering"""
    X_eng = X.copy()
    X_eng['N_P_ratio'] = X['N'] / (X['P'] + 1)
    X_eng['N_K_ratio'] = X['N'] / (X['K'] + 1)
    X_eng['P_K_ratio'] = X['P'] / (X['K'] + 1)
    X_eng['NPK_sum'] = X['N'] + X['P'] + X['K']
    X_eng['temp_humidity'] = X['temperature'] * X['humidity']
    X_eng['ph_acidic'] = (X['ph'] < 6.5).astype(int)
    X_eng['ph_alkaline'] = (X['ph'] > 7.5).astype(int)
    return X_eng

def load_model_and_data():
    """Load trained model and data"""
    
    print("="*70)
    print("LOADING MODEL AND DATA")
    print("="*70)
    
    with open('models/RF_MODEL_FOR_XAI.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    crop_names = model_package['crop_names']
    
    print(f"✅ Model loaded")
    print(f"   Crops: {len(crop_names)}")
    print(f"   Accuracy: {model_package['performance']['test_accuracy']*100:.2f}%")
    
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train_original = data['X_train']
    X_test_original = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"\n⚙️  Applying feature engineering...")
    X_train = engineer_features(X_train_original)
    X_test = engineer_features(X_test_original)
    
    print(f"✅ Data prepared")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Features: {len(X_train.columns)}")
    
    return model, X_train, X_test, y_train, y_test, crop_names

def generate_shap_explanations(model, X_train, X_test, crop_names):
    """Generate SHAP explanations"""
    
    print("\n" + "="*70)
    print("LAYER 1: SHAP - GLOBAL FEATURE IMPORTANCE")
    print("="*70)
    
    print("\n📊 Preparing SHAP explainer...")
    background = shap.sample(X_train, 100, random_state=42)
    
    start_time = time.time()
    explainer = shap.TreeExplainer(model, background)
    print(f"   ✅ Explainer created in {time.time() - start_time:.2f}s")
    
    print(f"\n🔄 Computing SHAP values...")
    test_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    
    start_time = time.time()
    shap_values = explainer.shap_values(test_sample, check_additivity=False)
    print(f"   ✅ SHAP values computed in {time.time() - start_time:.2f}s")
    
    return explainer, shap_values, test_sample

def analyze_global_importance(shap_values, test_sample):
    """Analyze global feature importance"""
    
    print("\n" + "="*70)
    print("GLOBAL FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    shap_array = np.array(shap_values)
    mean_abs_shap = np.abs(shap_array).mean(axis=(0, 1))
    
    importance_df = pd.DataFrame({
        'Feature': test_sample.columns,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    importance_df['Importance_Pct'] = (
        importance_df['Mean_Abs_SHAP'] / 
        importance_df['Mean_Abs_SHAP'].sum() * 100
    )
    
    print("\n🌟 Global Feature Importance (SHAP):")
    print("-"*70)
    for idx, row in importance_df.iterrows():
        bar_length = int(row['Importance_Pct'] / 2)
        bar = '█' * bar_length
        print(f"  {row['Feature']:20} {bar:30} {row['Importance_Pct']:5.2f}%")
    
    return importance_df

def create_dependence_plots(shap_values, test_sample, importance_df):
    """
    NEW: Create SHAP dependence plots showing feature interactions
    """
    
    print("\n" + "="*70)
    print("CREATING SHAP DEPENDENCE PLOTS (Feature Interactions)")
    print("="*70)
    
    os.makedirs('results/shap', exist_ok=True)
    
    # Get top 3 features
    top_features = importance_df.head(3)['Feature'].tolist()
    
    # Average SHAP values across classes
    shap_array = np.array(shap_values)
    mean_shap = shap_array.mean(axis=0)  # [n_samples, n_features]
    
    print(f"\nCreating dependence plots for top 3 features...")
    
    # Create 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, feature in enumerate(top_features):
        print(f"   {idx+1}. {feature}")
        
        feature_idx = list(test_sample.columns).index(feature)
        
        # Create dependence plot
        ax = axes[idx]
        
        # Get feature values and SHAP values
        feature_values = test_sample[feature].values
        shap_vals = mean_shap[:, feature_idx]
        
        # Scatter plot
        scatter = ax.scatter(feature_values, shap_vals, 
                           c=feature_values, cmap='viridis',
                           alpha=0.6, s=20)
        
        ax.set_xlabel(f'{feature} Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('SHAP Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Dependence: {feature}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label=f'{feature} Value')
    
    plt.tight_layout()
    plt.savefig('results/shap/dependence_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/shap/dependence_plots.png")
    plt.close()

def analyze_per_crop_importance(shap_values, test_sample, y_test_sample, crop_names, importance_df):
    """
    NEW: Analyze which features matter most for SPECIFIC crops
    """
    
    print("\n" + "="*70)
    print("PER-CROP SHAP ANALYSIS (Top 3 Crops)")
    print("="*70)
    
    # Get predictions for test samples
    # Find top 3 most common crops in test sample
    crop_counts = pd.Series(y_test_sample).value_counts()
    top_3_crops = crop_counts.head(3).index.tolist()
    
    print(f"\nAnalyzing top 3 crops by sample count:")
    for i, crop in enumerate(top_3_crops, 1):
        count = crop_counts[crop]
        print(f"   {i}. {crop} ({count} samples)")
    
    # Create per-crop importance analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    shap_array = np.array(shap_values)
    
    per_crop_results = []
    
    for idx, crop in enumerate(top_3_crops):
        print(f"\n   Analyzing {crop}...")
        
        # Get crop index
        crop_idx = crop_names.index(crop)
        
        # Get samples for this crop
        crop_mask = y_test_sample == crop
        crop_indices = np.where(crop_mask)[0]
        
        if len(crop_indices) > 0:
            # Get SHAP values for this crop's class
            crop_shap = shap_array[crop_idx, crop_indices, :]  # [n_samples, n_features]
            
            # Mean absolute SHAP for this crop
            crop_importance = np.abs(crop_shap).mean(axis=0)
            
            # Create DataFrame
            crop_importance_df = pd.DataFrame({
                'Feature': test_sample.columns,
                'Importance': crop_importance
            }).sort_values('Importance', ascending=False)
            
            # Normalize to percentage
            crop_importance_df['Percentage'] = (
                crop_importance_df['Importance'] / 
                crop_importance_df['Importance'].sum() * 100
            )
            
            per_crop_results.append({
                'crop': crop,
                'importance': crop_importance_df
            })
            
            # Plot
            ax = axes[idx]
            top_10 = crop_importance_df.head(10)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
            ax.barh(range(len(top_10)), top_10['Percentage'].values, 
                   color=colors, alpha=0.8)
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['Feature'].values)
            ax.set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{crop.upper()}', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/shap/per_crop_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/shap/per_crop_importance.png")
    plt.close()
    
    # Print comparison
    print(f"\n📊 Feature Importance Comparison Across Crops:")
    print("-"*70)
    
    comparison_data = []
    for result in per_crop_results:
        top_3 = result['importance'].head(3)
        comparison_data.append({
            'Crop': result['crop'],
            'Top_1': f"{top_3.iloc[0]['Feature']} ({top_3.iloc[0]['Percentage']:.1f}%)",
            'Top_2': f"{top_3.iloc[1]['Feature']} ({top_3.iloc[1]['Percentage']:.1f}%)",
            'Top_3': f"{top_3.iloc[2]['Feature']} ({top_3.iloc[2]['Percentage']:.1f}%)"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    return per_crop_results

def create_basic_visualizations(shap_values, test_sample):
    """Create standard SHAP visualizations"""
    
    print("\n" + "="*70)
    print("CREATING STANDARD SHAP VISUALIZATIONS")
    print("="*70)
    
    os.makedirs('results/shap', exist_ok=True)
    
    shap_array = np.array(shap_values)
    mean_shap = np.abs(shap_array).mean(axis=0)
    
    # 1. Feature importance bar
    print("\n1. Feature importance bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(mean_shap, test_sample, plot_type="bar", show=False)
    plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/shap/feature_importance_bar.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/shap/feature_importance_bar.png")
    plt.close()
    
    # 2. Summary plot
    print("\n2. Feature impact summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(mean_shap, test_sample, show=False)
    plt.title('Feature Impact on Model Output (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/shap/feature_impact_summary.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/shap/feature_impact_summary.png")
    plt.close()

def main():
    """Main execution with enhancements"""
    
    print("\n" + "🔍"*35)
    print("LAYER 1: ENHANCED SHAP ANALYSIS")
    print("🔍"*35)
    
    # Load
    model, X_train, X_test, y_train, y_test, crop_names = load_model_and_data()
    
    # Generate SHAP
    explainer, shap_values, test_sample = generate_shap_explanations(
        model, X_train, X_test, crop_names
    )
    
    # Get test labels for sampled data
    y_test_sample = y_test.loc[test_sample.index]
    
    # Analyze global importance
    importance_df = analyze_global_importance(shap_values, test_sample)
    
    # Create basic visualizations
    create_basic_visualizations(shap_values, test_sample)
    
    # ========================================
    # NEW: Dependence plots
    # ========================================
    create_dependence_plots(shap_values, test_sample, importance_df)
    
    # ========================================
    # NEW: Per-crop analysis
    # ========================================
    per_crop_results = analyze_per_crop_importance(
        shap_values, test_sample, y_test_sample, crop_names, importance_df
    )
    
    # Save results
    print("\n💾 Saving results...")
    importance_df.to_csv('results/shap/global_feature_importance.csv', index=False)
    print("   ✅ Saved: results/shap/global_feature_importance.csv")
    
    # Summary
    print("\n" + "="*70)
    print("✅ ENHANCED LAYER 1 (SHAP) COMPLETED!")
    print("="*70)
    
    print("\n📊 Key Findings:")
    print(f"   Top 3 Global Features:")
    for i, row in importance_df.head(3).iterrows():
        print(f"      {i+1}. {row['Feature']} ({row['Importance_Pct']:.2f}%)")
    
    print(f"\n📁 Generated Files:")
    print(f"   Standard:")
    print(f"   • results/shap/global_feature_importance.csv")
    print(f"   • results/shap/feature_importance_bar.png")
    print(f"   • results/shap/feature_impact_summary.png")
    print(f"   NEW Enhancements:")
    print(f"   • results/shap/dependence_plots.png          ⭐")
    print(f"   • results/shap/per_crop_importance.png       ⭐")
    
    print(f"\n🎯 Interpretation:")
    print(f"   - Global importance: What matters OVERALL")
    print(f"   - Dependence plots: HOW features affect predictions")
    print(f"   - Per-crop analysis: What matters for SPECIFIC crops")
    
    print(f"\n➡️  Next: Layer 2 (LIME)")
    
    return importance_df, per_crop_results

if __name__ == "__main__":
    importance_df, per_crop_results = main()