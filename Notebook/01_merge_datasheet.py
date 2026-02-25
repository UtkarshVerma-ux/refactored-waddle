import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def load_excel_dataset():
    """Load and preprocess Excel dataset"""
    
    print("="*70)
    print("LOADING EXCEL DATASET")
    print("="*70)
    
    df = pd.read_excel('../data/Bangladesh.xlsx')
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Standardize column names
    df = df.rename(columns={
        'Temp(°C)': 'temperature',
        'Humidity(%)': 'humidity',
        'Rainfall(cm)': 'rainfall',
        'pH': 'ph',
        'Crop': 'label'
    })
    
    # Drop Moisture% (not in Kaggle dataset)
    df = df.drop(columns=['Moisture(%)'])
    
    # ⚠️ CRITICAL: Convert rainfall from cm to mm (Kaggle standard)
    # Bangladesh dataset: 40-250 cm
    # Kaggle dataset: 400-2500 mm
    df['rainfall'] = df['rainfall'] * 10  # cm → mm
    
    print(f"  Rainfall converted: cm → mm (×10)")
    
    # Standardize crop names
    crop_mapping = {
        'Aman Rice': 'rice',
        'Aush Rice': 'rice',
        'Boro Rice': 'rice',
        'Corn': 'maize',
        'Bean': 'kidneybeans',
        'Pulse': 'lentil',
        'Peanut': 'groundnut',
        'Cauliflower': 'cauliflower',
        'Eggplant': 'eggplant',
        'Onion': 'onion',
        'Potato': 'potato',
        'Tomato': 'tomato',
        'Pumpkin': 'pumpkin',
        'Calabash': 'calabash',
        'Pointed Gourd': 'pointed_gourd',
        'Colocasia leaves': 'colocasia',
        'Leafy Veg': 'leafy_vegetables',
        'Pineapple': 'pineapple',
        'Oil crop': 'oil_crop',
        'Jute': 'jute',
        'Papaya': 'papaya',
        'Wheat': 'wheat'
    }
    
    df['label'] = df['label'].map(crop_mapping)
    
    # Reorder columns to match Kaggle format
    df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]
    
    print(f"\nAfter preprocessing:")
    print(f"  Shape: {df.shape}")
    print(f"  Unique crops: {df['label'].nunique()}")
    print(f"  Crop distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df

def load_kaggle_dataset():
    """Load Kaggle dataset (simulated - you'll need actual file)"""
    
    print("\n" + "="*70)
    print("LOADING KAGGLE DATASET")
    print("="*70)
    
    # Placeholder - will load actual data when available
    try:
        df = pd.read_csv('../data/Crop_recommendation.csv')
        print(f"✅ Loaded Kaggle dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Kaggle dataset not found!")
        print("   Continuing with Excel dataset only...")
        return None

def merge_datasets(excel_df, kaggle_df=None):
    """Merge both datasets"""
    
    print("\n" + "="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    if kaggle_df is None:
        print("Using Excel dataset only")
        merged_df = excel_df
    else:
        # Ensure both have same columns
        common_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        
        excel_subset = excel_df[common_cols]
        kaggle_subset = kaggle_df[common_cols]
        
        # Merge
        merged_df = pd.concat([excel_subset, kaggle_subset], ignore_index=True)
        
        print(f"Excel samples: {len(excel_df)}")
        print(f"Kaggle samples: {len(kaggle_df)}")
        print(f"Merged samples: {len(merged_df)}")
    
    print(f"\nFinal dataset:")
    print(f"  Total samples: {len(merged_df)}")
    print(f"  Unique crops: {merged_df['label'].nunique()}")
    print(f"\nCrop distribution:")
    print(merged_df['label'].value_counts().sort_index())
    
    # Check for data quality
    print(f"\nData quality:")
    print(f"  Missing values: {merged_df.isnull().sum().sum()}")
    print(f"  Duplicates: {merged_df.duplicated().sum()}")
    
    return merged_df

def prepare_final_dataset(merged_df):
    """Prepare final train-test split"""
    
    print("\n" + "="*70)
    print("PREPARING TRAIN-TEST SPLIT")
    print("="*70)
    
    # Features and target
    X = merged_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = merged_df['label']  # Keep as strings - NO ENCODING!
    
    # ✅ NO LabelEncoder - Random Forest handles strings directly!
    # This makes DiCE integration much cleaner
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Stratify on string labels
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Unique crops: {y.nunique()}")
    print(f"Label type: {type(y_train.iloc[0])} (keeping as strings for DiCE compatibility)")
    
    # Save processed data
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'crop_names': sorted(y.unique().tolist()),  # Alphabetically sorted crop list
        'full_dataset': merged_df
    }
    
    with open('data/processed_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    # Also save as CSV for easy viewing
    merged_df.to_csv('data/merged_dataset.csv', index=False)
    
    print(f"\n✅ Saved processed data to:")
    print(f"   - data/processed_data.pkl")
    print(f"   - data/merged_dataset.csv")
    print(f"\n💡 Labels kept as strings (not encoded) for easier DiCE integration!")
    
    return data

def main():
    """Main merge pipeline"""
    
    print("\n" + "🌾"*35)
    print("COMPREHENSIVE CROP DATASET MERGER")
    print("🌾"*35 + "\n")
    
    # Load Excel dataset
    excel_df = load_excel_dataset()
    
    # Try to load Kaggle dataset
    kaggle_df = load_kaggle_dataset()
    
    # Merge
    merged_df = merge_datasets(excel_df, kaggle_df)
    
    # Prepare final dataset
    data = prepare_final_dataset(merged_df)
    
    print("\n" + "="*70)
    print("✅ DATASET MERGE COMPLETE!")
    print("="*70)
    print(f"\nFinal Statistics:")
    print(f"  Total samples: {len(merged_df)}")
    print(f"  Unique crops: {len(data['crop_names'])}")
    print(f"  Features: {len(data['feature_names'])}")
    print(f"  Train/Test: {len(data['X_train'])}/{len(data['X_test'])}")
    
    print(f"\nCrop list ({len(data['crop_names'])} crops):")
    for i, crop in enumerate(sorted(data['crop_names']), 1):
        print(f"  {i:2}. {crop}")
    
    print("\n" + "🌾"*35 + "\n")

if __name__ == "__main__":
    main()