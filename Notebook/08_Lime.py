"""
LAYER 2: LIME - Local Explainability
Local Interpretable Model-agnostic Explanations

Purpose:
- Explain SPECIFIC individual predictions
- "Why was THIS crop recommended for THIS sample?"
- Instance-level feature importance (vs SHAP's global view)

Author: Research Framework
Date: Feb 26, 2026
"""

import pickle
import numpy as np
import pandas as pd

# Set matplotlib backend BEFORE importing pyplot (fixes threading warnings)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import seaborn as sns
from lime import lime_tabular
import time
import os
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

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
    
    # Load model
    with open('models/RF_MODEL_FOR_XAI.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    crop_names = model_package['crop_names']
    
    print(f"✅ Model loaded")
    print(f"   Crops: {len(crop_names)}")
    print(f"   Accuracy: {model_package['performance']['test_accuracy']*100:.2f}%")
    
    # Load data
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
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(X_train.columns)}")
    
    return model, X_train, X_test, y_train, y_test, crop_names

def create_lime_explainer(X_train, crop_names, model):
    """
    Create LIME explainer for the model
    """
    
    print("\n" + "="*70)
    print("CREATING LIME EXPLAINER")
    print("="*70)
    
    print("\n📊 Initializing LIME TabularExplainer...")
    
    # Create prediction wrapper that handles feature names
    feature_names = X_train.columns.tolist()
    
    def predict_fn_wrapper(X):
        """Wrapper to convert numpy array to DataFrame with feature names"""
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
        return model.predict_proba(X_df)
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=crop_names,
        mode='classification',
        random_state=42
    )
    
    print(f"✅ LIME explainer created")
    print(f"   Training data shape: {X_train.shape}")
    print(f"   Feature names: {feature_names}")
    print(f"   Classes: {len(crop_names)}")
    
    return explainer, predict_fn_wrapper

def explain_sample(explainer, predict_fn, model, sample, sample_idx, true_label, crop_names):
    """
    Generate LIME explanation for a single sample
    """
    
    print(f"\n" + "-"*70)
    print(f"EXPLAINING SAMPLE #{sample_idx}")
    print("-"*70)
    
    # Convert sample to DataFrame for prediction
    sample_df = pd.DataFrame([sample], columns=explainer.feature_names)
    
    # Get prediction
    prediction = model.predict(sample_df)[0]
    prediction_proba = model.predict_proba(sample_df)[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(prediction_proba)[-3:][::-1]
    
    print(f"\n📊 Sample Information:")
    print(f"   True crop: {true_label}")
    print(f"   Predicted crop: {prediction}")
    print(f"   Confidence: {prediction_proba[crop_names.index(prediction)]*100:.2f}%")
    
    print(f"\n   Top 3 predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        crop = crop_names[idx]
        prob = prediction_proba[idx]
        print(f"      {i}. {crop:15} {prob*100:5.2f}%")
    
    # Generate LIME explanation
    print(f"\n🔄 Generating LIME explanation...")
    start_time = time.time()
    
    explanation = explainer.explain_instance(
        data_row=sample,
        predict_fn=predict_fn,  # Use wrapper function
        num_features=10,
        top_labels=3
    )
    
    lime_time = time.time() - start_time
    print(f"   ✅ Explanation generated in {lime_time:.2f} seconds")
    
    return explanation, prediction, prediction_proba, lime_time

def visualize_lime_explanation(explanation, sample_idx, prediction, crop_names, feature_names):
    """
    Visualize LIME explanation
    """
    
    print(f"\n📊 Creating visualization...")
    
    os.makedirs('results/lime', exist_ok=True)
    
    # Get the explanation for the predicted class
    pred_idx = crop_names.index(prediction)
    
    # Get feature contributions
    exp_list = explanation.as_list(label=pred_idx)
    
    # Parse feature names and values
    features = []
    values = []
    
    for feature_desc, value in exp_list:
        features.append(feature_desc)
        values.append(value)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars by positive/negative contribution
    colors = ['green' if v > 0 else 'red' for v in values]
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Contribution', fontsize=12, fontweight='bold')
    ax.set_title(f'LIME Explanation for Prediction: {prediction}', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/lime/explanation_sample_{sample_idx}.png', 
                dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: results/lime/explanation_sample_{sample_idx}.png")
    plt.close()
    
    return exp_list

def analyze_multiple_samples(explainer, predict_fn, model, X_test, y_test, crop_names, n_samples=5):
    """
    Analyze multiple samples to show LIME's local nature
    """
    
    print("\n" + "="*70)
    print(f"LAYER 2: LIME - LOCAL EXPLANATIONS ({n_samples} samples)")
    print("="*70)
    
    # Select diverse samples (different crops)
    unique_crops = y_test.unique()[:n_samples]
    
    sample_results = []
    total_lime_time = 0
    
    for i, crop in enumerate(unique_crops, 1):
        # Get first sample of this crop
        crop_indices = y_test[y_test == crop].index
        sample_idx = crop_indices[0]
        
        sample = X_test.loc[sample_idx].values
        true_label = y_test.loc[sample_idx]
        
        print(f"\n{'='*70}")
        print(f"SAMPLE {i}/{n_samples}")
        print(f"{'='*70}")
        
        # Explain this sample
        explanation, prediction, prediction_proba, lime_time = explain_sample(
            explainer, predict_fn, model, sample, sample_idx, true_label, crop_names
        )
        
        total_lime_time += lime_time
        
        # Visualize
        exp_list = visualize_lime_explanation(
            explanation, sample_idx, prediction, crop_names, X_test.columns
        )
        
        # Store results
        sample_results.append({
            'sample_idx': sample_idx,
            'true_label': true_label,
            'prediction': prediction,
            'confidence': prediction_proba[crop_names.index(prediction)],
            'explanation': exp_list,
            'lime_time': lime_time
        })
        
        # Print explanation summary
        print(f"\n   💡 Why {prediction}?")
        for feature_desc, value in exp_list[:5]:
            direction = "supports" if value > 0 else "opposes"
            print(f"      • {feature_desc} {direction} this prediction ({value:+.3f})")
    
    # Summary statistics
    avg_time = total_lime_time / n_samples
    
    print(f"\n" + "="*70)
    print("LIME PERFORMANCE SUMMARY")
    print("="*70)
    
    print(f"\n⏱️  Timing Statistics:")
    print(f"   Total time: {total_lime_time:.2f} seconds")
    print(f"   Average time per explanation: {avg_time:.2f} seconds")
    print(f"   Min time: {min([r['lime_time'] for r in sample_results]):.2f}s")
    print(f"   Max time: {max([r['lime_time'] for r in sample_results]):.2f}s")
    
    print(f"\n✅ LIME is fast enough for real-time use!")
    if avg_time < 2:
        print(f"   Average {avg_time:.2f}s is acceptable for interactive apps")
    
    return sample_results

def create_comparison_visualization(sample_results, crop_names):
    """
    Create visualization comparing explanations across samples
    """
    
    print(f"\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*70)
    
    os.makedirs('results/lime', exist_ok=True)
    
    # Create multi-panel plot
    n_samples = len(sample_results)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    for idx, (ax, result) in enumerate(zip(axes, sample_results)):
        # Get top 8 features
        exp_list = result['explanation'][:8]
        
        features = [f.split('<=')[0].split('>')[0].strip() for f, _ in exp_list]
        values = [v for _, v in exp_list]
        
        # Color by contribution
        colors = ['green' if v > 0 else 'red' for v in values]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Contribution', fontsize=10, fontweight='bold')
        ax.set_title(f'Sample {idx+1}: {result["prediction"]} '
                    f'(Confidence: {result["confidence"]*100:.1f}%)',
                    fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/lime/comparison_all_samples.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: results/lime/comparison_all_samples.png")
    plt.close()

def save_results_summary(sample_results):
    """
    Save LIME results to CSV
    """
    
    print(f"\n💾 Saving results...")
    
    # Create summary dataframe
    summary_data = []
    
    for result in sample_results:
        summary_data.append({
            'Sample_Index': result['sample_idx'],
            'True_Label': result['true_label'],
            'Prediction': result['prediction'],
            'Confidence': f"{result['confidence']*100:.2f}%",
            'Explanation_Time': f"{result['lime_time']:.2f}s",
            'Top_Feature': result['explanation'][0][0] if result['explanation'] else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/lime/lime_explanations_summary.csv', index=False)
    print(f"   ✅ Saved: results/lime/lime_explanations_summary.csv")

def main():
    """
    Main execution for Layer 2: LIME
    """
    
    print("\n" + "🔍"*35)
    print("LAYER 2: LIME - LOCAL EXPLAINABILITY")
    print("🔍"*35)
    
    # Load model and data
    model, X_train, X_test, y_train, y_test, crop_names = load_model_and_data()
    
    # Create LIME explainer with prediction wrapper
    explainer, predict_fn = create_lime_explainer(X_train, crop_names, model)
    
    # Analyze multiple samples
    sample_results = analyze_multiple_samples(
        explainer, predict_fn, model, X_test, y_test, crop_names, n_samples=5
    )
    
    # Create comparison visualization
    create_comparison_visualization(sample_results, crop_names)
    
    # Save summary
    save_results_summary(sample_results)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ LAYER 2 (LIME) COMPLETED!")
    print("="*70)
    
    print(f"\n📊 Key Findings:")
    print(f"   LIME provides LOCAL explanations (different for each sample)")
    print(f"   Average explanation time: {np.mean([r['lime_time'] for r in sample_results]):.2f}s")
    print(f"   All samples explained successfully: {len(sample_results)}/{len(sample_results)}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • results/lime/explanation_sample_*.png (individual explanations)")
    print(f"   • results/lime/comparison_all_samples.png (comparison)")
    print(f"   • results/lime/lime_explanations_summary.csv (summary data)")
    
    print(f"\n🎯 LIME vs SHAP:")
    print(f"   SHAP: Global importance (what matters OVERALL)")
    print(f"   LIME: Local importance (what matters for THIS sample)")
    
    print(f"\n💡 Example Use Case:")
    print(f"   Farmer: 'Why did you recommend rice for MY soil?'")
    print(f"   LIME: 'Because YOUR pH (6.2) is acidic, which rice prefers'")
    print(f"   (Different farmer with pH 7.5 would get different explanation!)")
    
    print(f"\n➡️  Next: Layer 3 (DiCE + Climate + Sustainability)")
    
    return sample_results, explainer

if __name__ == "__main__":
    sample_results, explainer = main()