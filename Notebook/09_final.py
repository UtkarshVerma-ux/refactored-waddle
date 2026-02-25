import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import time

# ============================================================================
# FEATURE ENGINEERING
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

# ============================================================================
# CROP CLIMATE DATABASE
# ============================================================================

# ============================================================================
# CROP CLIMATE DATABASE - LOADED FROM CSV
# ============================================================================

def load_crop_climate_database():
    """
    Load crop climate requirements from CSV file
    Returns: Dictionary with all crop requirements
    """
    import os
    
    csv_path = 'data/crop_climate_requirements.csv'
    
    if not os.path.exists(csv_path):
        print(f"⚠️  Warning: Climate database CSV not found at {csv_path}")
        print(f"   Using default requirements for all crops")
        return {}
    
    # Load CSV
    climate_df = pd.read_csv(csv_path)
    
    # Convert to dictionary format
    crop_requirements = {}
    
    for _, row in climate_df.iterrows():
        crop = row['crop']
        crop_requirements[crop] = {
            'temperature': {
                'min': row['temp_min'],
                'max': row['temp_max'],
                'optimal': (row['temp_opt_min'], row['temp_opt_max'])
            },
            'humidity': {
                'min': row['humidity_min'],
                'max': row['humidity_max'],
                'optimal': (row['humidity_opt_min'], row['humidity_opt_max'])
            },
            'rainfall': {
                'min': row['rainfall_min'],
                'max': row['rainfall_max'],
                'optimal': (row['rainfall_opt_min'], row['rainfall_opt_max'])
            },
            'ph': {
                'min': row['ph_min'],
                'max': row['ph_max'],
                'optimal': (row['ph_opt_min'], row['ph_opt_max'])
            }
        }
    
    print(f"✅ Loaded climate requirements for {len(crop_requirements)} crops from CSV")
    return crop_requirements

# Load database on import
CROP_CLIMATE_DATABASE = load_crop_climate_database()

def get_crop_requirements(crop):
    """
    Get climate requirements for a crop
    
    Raises error if crop not in database to catch data issues early
    """
    if crop in CROP_CLIMATE_DATABASE:
        return CROP_CLIMATE_DATABASE[crop]
    else:
        # This should never happen if CSV has all crops!
        available_crops = ', '.join(sorted(CROP_CLIMATE_DATABASE.keys())[:10])
        raise ValueError(
            f"❌ Crop '{crop}' not found in climate database!\n"
            f"   Available crops: {available_crops}...\n"
            f"   Please add '{crop}' to data/crop_climate_requirements.csv"
        )

# ============================================================================
# LAYER 3A: CLIMATE FEASIBILITY CHECK
# ============================================================================

def check_climate_feasibility(sample, crop):
    """
    Check if crop can grow in given climate conditions
    Returns: (feasible: bool, reasons: dict, score: float)
    """
    
    requirements = get_crop_requirements(crop)
    
    # Extract climate features
    temp = sample['temperature']
    humidity = sample['humidity']
    rainfall = sample['rainfall']
    ph = sample.get('ph', 6.5)  # Use pH if available
    
    # Check each constraint
    checks = {}
    
    # Temperature
    temp_ok = requirements['temperature']['min'] <= temp <= requirements['temperature']['max']
    checks['temperature'] = {
        'feasible': temp_ok,
        'current': temp,
        'required': f"{requirements['temperature']['min']}-{requirements['temperature']['max']}°C"
    }
    
    # Humidity
    hum_ok = requirements['humidity']['min'] <= humidity <= requirements['humidity']['max']
    checks['humidity'] = {
        'feasible': hum_ok,
        'current': humidity,
        'required': f"{requirements['humidity']['min']}-{requirements['humidity']['max']}%"
    }
    
    # Rainfall
    rain_ok = requirements['rainfall']['min'] <= rainfall <= requirements['rainfall']['max']
    checks['rainfall'] = {
        'feasible': rain_ok,
        'current': rainfall,
        'required': f"{requirements['rainfall']['min']}-{requirements['rainfall']['max']}mm"
    }
    
    # Overall feasibility
    overall_feasible = temp_ok and hum_ok and rain_ok
    
    # Calculate feasibility score (0-100)
    score = 0
    if temp_ok:
        score += 40
    if hum_ok:
        score += 30
    if rain_ok:
        score += 30
    
    return overall_feasible, checks, score

def filter_feasible_crops(sample, all_crops):
    """
    Filter crops that can grow in given climate
    Returns list of feasible crops with scores
    """
    
    print(f"\n🌦️  LAYER 3A: Climate Feasibility Check")
    print("-"*70)
    
    print(f"\n   Current climate:")
    print(f"      Temperature: {sample['temperature']:.1f}°C")
    print(f"      Humidity: {sample['humidity']:.1f}%")
    print(f"      Rainfall: {sample['rainfall']:.0f}mm")
    
    feasible_crops = []
    infeasible_crops = []
    
    for crop in all_crops:
        feasible, checks, score = check_climate_feasibility(sample, crop)
        
        if feasible:
            feasible_crops.append({
                'crop': crop,
                'score': score,
                'checks': checks
            })
        else:
            reasons = []
            for param, check in checks.items():
                if not check['feasible']:
                    reasons.append(f"{param}: {check['current']} (need {check['required']})")
            infeasible_crops.append({
                'crop': crop,
                'reasons': reasons
            })
    
    print(f"\n   ✅ Feasible crops: {len(feasible_crops)}/{len(all_crops)}")
    print(f"   ❌ Infeasible crops: {len(infeasible_crops)}/{len(all_crops)}")
    
    # Show first 5 feasible and infeasible
    if feasible_crops:
        print(f"\n   Sample feasible crops:")
        for item in feasible_crops[:5]:
            print(f"      ✅ {item['crop']:15} (climate score: {item['score']}/100)")
    
    if infeasible_crops:
        print(f"\n   Sample infeasible crops:")
        for item in infeasible_crops[:3]:
            print(f"      ❌ {item['crop']:15} ({', '.join(item['reasons'][:2])})")
    
    return feasible_crops, infeasible_crops

# ============================================================================
# LAYER 3B: CONSTRAINT-AWARE DiCE (GENETIC ALGORITHM)
# ============================================================================

def generate_dice_counterfactual(model, current_sample, target_crop, 
                                 controllable_features=['N', 'P', 'K', 'ph'],
                                 n_iterations=100):  # Increased from 50
    """
    Generate counterfactual using genetic algorithm
    Only varies controllable features (N, P, K, pH)
    Keeps climate features fixed (uncontrollable)
    """
    
    print(f"\n🔄 LAYER 3B: DiCE Counterfactual Generation")
    print("-"*70)
    print(f"   Target crop: {target_crop}")
    print(f"   Controllable features: {controllable_features}")
    
    # Define feature ranges
    feature_ranges = {
        'N': (0, 140),
        'P': (5, 145),
        'K': (5, 205),
        'ph': (3.5, 9.5)
    }
    
    # Extract current values
    current_values = {feat: current_sample[feat] for feat in controllable_features}
    
    print(f"\n   Current values:")
    for feat, val in current_values.items():
        print(f"      {feat}: {val:.1f}")
    
    # Initialize population (100 candidates for better diversity)
    population_size = 100  # Increased from 50
    population = []
    
    for _ in range(population_size):
        candidate = current_sample.copy()
        # Randomly modify controllable features with larger perturbations
        for feat in controllable_features:
            min_val, max_val = feature_ranges[feat]
            # Random perturbation within ±50% of range (increased from 20%)
            if np.random.random() < 0.7:  # 70% of time, stay near current
                delta = np.random.uniform(-0.3, 0.3) * current_values[feat]
                new_val = np.clip(current_values[feat] + delta, min_val, max_val)
            else:  # 30% of time, explore full range
                new_val = np.random.uniform(min_val, max_val)
            candidate[feat] = new_val
        population.append(candidate)
    
    best_candidate = None
    best_probability = 0
    
    # Genetic algorithm iterations
    start_time = time.time()
    
    for iteration in range(n_iterations):
        # Evaluate population
        scores = []
        
        for candidate in population:
            # Engineer features
            candidate_df = pd.DataFrame([candidate])
            candidate_eng = engineer_features(candidate_df)
            
            # Predict
            proba = model.predict_proba(candidate_eng)[0]
            crop_idx = model.classes_.tolist().index(target_crop)
            target_prob = proba[crop_idx]
            
            # Calculate distance from current (prefer minimal changes)
            distance = sum((candidate[f] - current_values[f])**2 for f in controllable_features)
            
            # Fitness = target probability - distance penalty
            fitness = target_prob - 0.001 * distance
            scores.append(fitness)
            
            # Track best
            if target_prob > best_probability:
                best_probability = target_prob
                best_candidate = candidate.copy()
        
        # Selection: keep top 50%
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx[:population_size//2]]
        
        # Crossover & mutation: generate new candidates
        while len(population) < population_size:
            # Random crossover
            parent1, parent2 = np.random.choice(population, 2, replace=False)
            child = current_sample.copy()
            
            for feat in controllable_features:
                # Crossover
                child[feat] = np.random.choice([parent1[feat], parent2[feat]])
                
                # Mutation (10% chance)
                if np.random.random() < 0.1:
                    min_val, max_val = feature_ranges[feat]
                    delta = np.random.uniform(-0.1, 0.1) * current_values[feat]
                    child[feat] = np.clip(child[feat] + delta, min_val, max_val)
            
            population.append(child)
    
    dice_time = time.time() - start_time
    
    # Check if we found a good counterfactual
    if best_candidate is None or best_probability < 0.1:
        print(f"\n   ⚠️  Could not find good counterfactual in {dice_time:.2f} seconds")
        print(f"   Best probability achieved: {best_probability*100:.1f}%")
        print(f"   Note: This crop may not be achievable with current climate")
        
        # Return current values (no changes recommended)
        return current_sample.copy(), best_probability, dice_time
    
    print(f"\n   ✅ Counterfactual found in {dice_time:.2f} seconds")
    print(f"   Target probability: {best_probability*100:.1f}%")
    
    print(f"\n   Recommended changes:")
    for feat in controllable_features:
        change = best_candidate[feat] - current_values[feat]
        print(f"      {feat}: {current_values[feat]:.1f} → {best_candidate[feat]:.1f} ({change:+.1f})")
    
    return best_candidate, best_probability, dice_time

# ============================================================================
# LAYER 3C: SUSTAINABILITY SCORING
# ============================================================================

def calculate_sustainability_score(current, counterfactual):
    """
    Calculate multi-dimensional sustainability score (0-100)
    
    Components:
    1. Chemical use (35%): Less N, P, K is better
    2. Soil health (25%): Balanced NPK ratio, optimal pH
    3. Water conservation (20%): Lower water needs
    4. Carbon footprint (20%): Lower overall inputs
    """
    
    print(f"\n🌱 LAYER 3C: Sustainability Scoring")
    print("-"*70)
    
    scores = {}
    
    # 1. Chemical use score (35%)
    # Lower NPK usage = higher score
    current_npk = current['N'] + current['P'] + current['K']
    cf_npk = counterfactual['N'] + counterfactual['P'] + counterfactual['K']
    
    if cf_npk <= current_npk:
        chemical_score = 100  # No increase
    else:
        # Penalize based on percentage increase
        increase = (cf_npk - current_npk) / current_npk
        chemical_score = max(0, 100 - increase * 100)
    
    scores['chemical_use'] = chemical_score
    print(f"   Chemical use score: {chemical_score:.1f}/100")
    print(f"      Current NPK: {current_npk:.0f} kg/ha")
    print(f"      New NPK: {cf_npk:.0f} kg/ha")
    
    # 2. Soil health score (25%)
    # Balanced NPK ratio and optimal pH
    
    # NPK balance (ideal ratio ~4:2:1 for N:P:K)
    cf_n_p_ratio = counterfactual['N'] / (counterfactual['P'] + 1)
    cf_n_k_ratio = counterfactual['N'] / (counterfactual['K'] + 1)
    
    ideal_n_p = 2.0
    ideal_n_k = 4.0
    
    ratio_score = (
        100 - abs(cf_n_p_ratio - ideal_n_p) * 20 +
        100 - abs(cf_n_k_ratio - ideal_n_k) * 10
    ) / 2
    ratio_score = np.clip(ratio_score, 0, 100)
    
    # pH score (6.5-7.0 is ideal)
    cf_ph = counterfactual['ph']
    if 6.5 <= cf_ph <= 7.0:
        ph_score = 100
    else:
        ph_score = max(0, 100 - abs(cf_ph - 6.75) * 30)
    
    soil_health = (ratio_score + ph_score) / 2
    scores['soil_health'] = soil_health
    print(f"   Soil health score: {soil_health:.1f}/100")
    print(f"      NPK balance: {ratio_score:.1f}/100")
    print(f"      pH optimality: {ph_score:.1f}/100 (pH={cf_ph:.1f})")
    
    # 3. Water conservation score (20%)
    # Based on crop water requirements (lower is better)
    # Proxy: rainfall requirement
    rainfall = current.get('rainfall', 1000)
    
    if rainfall < 500:
        water_score = 100  # Drought-tolerant region
    elif rainfall < 1000:
        water_score = 80
    elif rainfall < 1500:
        water_score = 60
    else:
        water_score = 40  # High water region
    
    scores['water_conservation'] = water_score
    print(f"   Water conservation: {water_score:.1f}/100")
    print(f"      Rainfall: {rainfall:.0f}mm")
    
    # 4. Carbon footprint score (20%)
    # Based on total inputs (fertilizer production has carbon cost)
    
    # Nitrogen has highest carbon footprint
    n_footprint = counterfactual['N'] * 2.0  # Roughly 2kg CO2 per kg N
    p_footprint = counterfactual['P'] * 1.5
    k_footprint = counterfactual['K'] * 0.5
    
    total_footprint = n_footprint + p_footprint + k_footprint
    
    # Lower footprint = higher score
    if total_footprint < 200:
        carbon_score = 100
    elif total_footprint < 400:
        carbon_score = 80
    elif total_footprint < 600:
        carbon_score = 60
    else:
        carbon_score = max(0, 100 - (total_footprint - 600) * 0.1)
    
    scores['carbon_footprint'] = carbon_score
    print(f"   Carbon footprint: {carbon_score:.1f}/100")
    print(f"      Estimated: {total_footprint:.0f} kg CO2e")
    
    # Overall score (weighted average)
    weights = {
        'chemical_use': 0.35,
        'soil_health': 0.25,
        'water_conservation': 0.20,
        'carbon_footprint': 0.20
    }
    
    overall = sum(scores[component] * weights[component] 
                  for component in scores.keys())
    
    print(f"\n   📊 Overall Sustainability: {overall:.1f}/100")
    
    return overall, scores

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_model_and_data():
    """Load model and data"""
    
    print("="*70)
    print("LOADING MODEL AND DATA")
    print("="*70)
    
    with open('models/RF_MODEL_FOR_XAI.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    crop_names = model_package['crop_names']
    
    print(f"✅ Model loaded: {len(crop_names)} crops")
    
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = engineer_features(data['X_test'])
    y_test = data['y_test']
    
    print(f"✅ Data loaded: {len(X_test)} test samples")
    
    return model, X_test, y_test, crop_names

def demonstrate_layer3(model, X_test, y_test, crop_names, sample_idx=0):
    """
    Demonstrate complete Layer 3 framework on a sample
    
    Args:
        sample_idx: Index value (not position) of the sample to use
    """
    
    print("\n" + "🌾"*35)
    print("LAYER 3: COMPLETE ACTIONABLE RECOMMENDATIONS")
    print("🌾"*35)
    
    # Get sample using .loc (index value) not .iloc (position)
    sample = X_test.loc[sample_idx]
    true_crop = y_test.loc[sample_idx]
    
    print(f"\n📍 Sample Information:")
    print(f"   Sample index: {sample_idx}")
    print(f"   True crop: {true_crop}")
    print(f"   Soil: N={sample['N']:.0f}, P={sample['P']:.0f}, K={sample['K']:.0f}, pH={sample['ph']:.1f}")
    print(f"   Climate: T={sample['temperature']:.1f}°C, H={sample['humidity']:.0f}%, R={sample['rainfall']:.0f}mm")
    
    # Current prediction
    sample_df = pd.DataFrame([sample])
    current_pred = model.predict(sample_df)[0]
    current_proba = model.predict_proba(sample_df)[0]
    
    print(f"\n   Current prediction: {current_pred} ({current_proba[crop_names.index(current_pred)]*100:.1f}%)")
    
    # ========================================
    # LAYER 3A: Climate Feasibility
    # ========================================
    
    feasible_crops, infeasible_crops = filter_feasible_crops(sample, crop_names)
    
    if len(feasible_crops) == 0:
        print("\n   ❌ No crops can grow in this climate!")
        return
    
    # ========================================
    # LAYER 3B + 3C: For top 3 feasible crops
    # ========================================
    
    # Sort feasible crops by climate score
    feasible_crops.sort(key=lambda x: x['score'], reverse=True)
    
    recommendations = []
    
    print(f"\n{'='*70}")
    print(f"GENERATING RECOMMENDATIONS FOR TOP 3 FEASIBLE CROPS")
    print(f"{'='*70}")
    
    for i, crop_info in enumerate(feasible_crops[:3], 1):
        target_crop = crop_info['crop']
        
        print(f"\n{'─'*70}")
        print(f"RECOMMENDATION {i}: {target_crop.upper()}")
        print(f"{'─'*70}")
        
        # Generate counterfactual
        counterfactual, probability, dice_time = generate_dice_counterfactual(
            model, sample.to_dict(), target_crop
        )
        
        # Skip if probability too low (counterfactual failed)
        if probability < 0.1:
            print(f"\n   ⚠️  Skipping {target_crop} - cannot achieve >10% probability")
            print(f"   This crop may not be viable with current climate constraints")
            continue
        
        # Calculate sustainability
        sustainability, components = calculate_sustainability_score(
            sample.to_dict(), counterfactual
        )
        
        # Estimate cost (simplified)
        n_change = abs(counterfactual['N'] - sample['N'])
        p_change = abs(counterfactual['P'] - sample['P'])
        k_change = abs(counterfactual['K'] - sample['K'])
        
        # Rough costs: N=$2/kg, P=$3/kg, K=$1.5/kg, lime=$15/unit pH change
        cost = n_change * 2 + p_change * 3 + k_change * 1.5
        if abs(counterfactual['ph'] - sample['ph']) > 0:
            cost += abs(counterfactual['ph'] - sample['ph']) * 15
        
        recommendations.append({
            'rank': i,
            'crop': target_crop,
            'probability': probability,
            'sustainability': sustainability,
            'cost': cost,
            'changes': {
                'N': counterfactual['N'] - sample['N'],
                'P': counterfactual['P'] - sample['P'],
                'K': counterfactual['K'] - sample['K'],
                'ph': counterfactual['ph'] - sample['ph']
            },
            'climate_score': crop_info['score'],
            'dice_time': dice_time
        })
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    
    if len(recommendations) == 0:
        print(f"\n" + "="*70)
        print("⚠️  NO VIABLE RECOMMENDATIONS FOUND")
        print("="*70)
        print(f"\n   The current climate conditions make it difficult to achieve")
        print(f"   high success probability for the climatically feasible crops.")
        print(f"   This may indicate:")
        print(f"   • Climate is at edge of tolerance for these crops")
        print(f"   • Soil changes alone insufficient (climate dominates)")
        print(f"   • Consider irrigation or greenhouse cultivation")
        return recommendations
    
    print(f"\n" + "="*70)
    print("📊 FINAL RECOMMENDATIONS SUMMARY")
    print("="*70)
    
    # Create summary table
    print(f"\n{'Rank':<6}{'Crop':<15}{'Success':<10}{'Sustain.':<10}{'Cost':<10}{'Climate':<10}")
    print("-"*70)
    
    for rec in recommendations:
        print(f"{rec['rank']:<6}{rec['crop']:<15}{rec['probability']*100:<10.1f}"
              f"{rec['sustainability']:<10.1f}${rec['cost']:<9.0f}{rec['climate_score']:<10.0f}")
    
    # Best overall (weighted score)
    for rec in recommendations:
        rec['overall_score'] = (
            rec['probability'] * 0.3 +
            rec['sustainability'] / 100 * 0.4 +
            rec['climate_score'] / 100 * 0.3
        )
    
    best_rec = max(recommendations, key=lambda x: x['overall_score'])
    
    print(f"\n🏆 BEST RECOMMENDATION: {best_rec['crop'].upper()}")
    print(f"   Overall score: {best_rec['overall_score']*100:.1f}/100")
    print(f"   Success probability: {best_rec['probability']*100:.1f}%")
    print(f"   Sustainability: {best_rec['sustainability']:.1f}/100")
    print(f"   Estimated cost: ${best_rec['cost']:.0f}")
    
    print(f"\n   Required actions:")
    for feat, change in best_rec['changes'].items():
        if abs(change) > 0.1:
            action = "Add" if change > 0 else "Reduce"
            print(f"      • {action} {feat}: {abs(change):.1f} units")
    
    return recommendations

def main():
    """Main execution"""
    
    print("\n" + "🌱"*35)
    print("LAYER 3: COMPLETE FRAMEWORK")
    print("Climate + DiCE + Sustainability")
    print("🌱"*35)
    
    # Load
    model, X_test, y_test, crop_names = load_model_and_data()
    
    # Find a sample with moderate rainfall (500-1500mm) for better demonstration
    good_samples = X_test[(X_test['rainfall'] > 500) & (X_test['rainfall'] < 1500)]
    if len(good_samples) > 0:
        sample_idx = good_samples.index[0]  # This is an INDEX VALUE (e.g., 3370)
        print(f"\n📍 Selected sample #{sample_idx} with moderate rainfall for demonstration")
    else:
        sample_idx = X_test.index[0]  # Fallback to first sample's index
    
    # Demonstrate on selected sample
    recommendations = demonstrate_layer3(model, X_test, y_test, crop_names, sample_idx=sample_idx)
    
    print(f"\n" + "="*70)
    print("✅ LAYER 3 COMPLETED!")
    print("="*70)
    
    print(f"\n🎯 Novel Contributions:")
    print(f"   ✅ Climate feasibility filtering (before DiCE)")
    print(f"   ✅ Constraint-aware DiCE (only controllable features)")
    print(f"   ✅ Multi-dimensional sustainability scoring")
    print(f"   ✅ Cost-aware recommendations")
    
    print(f"\n📝 For Paper:")
    print(f"   This framework bridges the gap between ML predictions")
    print(f"   and practical agricultural decision-making by:")
    print(f"   1. Validating climate feasibility")
    print(f"   2. Suggesting ACTIONABLE soil changes")
    print(f"   3. Scoring environmental sustainability")
    
    return recommendations

if __name__ == "__main__":
    recommendations = main()