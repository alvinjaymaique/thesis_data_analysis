import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
TQDM_AVAILABLE = True

# 1. Load all filtered anomalous data files
def load_anomalous_data(data_dir='filtered_data'):
    """Load and combine all filtered anomalous data files, adding location info."""
    all_files = glob.glob(f'{data_dir}/*_filtered.csv')
    data_frames = []
    
    for file_path in all_files:
        try:
            location = os.path.basename(file_path).replace('_filtered.csv', '')
            df = pd.read_csv(file_path)
            df['location'] = location
            data_frames.append(df)
            print(f"Loaded {location} - {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"Total combined data: {len(combined_df)} rows")
    return combined_df

# 2. Preprocess and clean data
def preprocess_data(df):
    """Handle missing values, standardize columns, and prepare data."""
    print("\nPreprocessing data...")
    df = df.copy()
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Ensure critical columns exist
    critical_cols = ['voltage', 'current', 'frequency', 'power', 'powerFactor']
    for col in critical_cols:
        if col not in df.columns:
            raise ValueError(f"Critical column '{col}' missing from data")
    
    # Handle missing values
    print(f"Missing values before: {df[critical_cols].isna().sum().sum()}")
    
    # Group by location and interpolate within each location
    df = df.groupby('location').apply(lambda x: x.interpolate(method='linear'))
    
    # If any remaining NaNs, fill with median values by location
    for loc in df['location'].unique():
        loc_mask = df['location'] == loc
        for col in critical_cols:
            median_val = df.loc[loc_mask, col].median()
            df.loc[loc_mask, col] = df.loc[loc_mask, col].fillna(median_val)
    
    # Drop any rows that still have NaN in critical columns
    df = df.dropna(subset=critical_cols)
    print(f"Missing values after: {df[critical_cols].isna().sum().sum()}")
    print(f"Rows after handling missing values: {len(df)}")
    
    return df

# 3. Feature Engineering
def engineer_features(df):
    """Create detailed features to help distinguish anomaly types."""
    print("\nEngineering features...")
    df = df.copy()
    
    # Define nominal/ideal values for Philippines power systems
    NOMINAL_VOLTAGE = 230.0    # Nominal voltage for Philippines (V)
    IDEAL_FREQUENCY = 60.0     # Ideal frequency (Hz)
    IDEAL_POWER_FACTOR = 1.0   # Ideal power factor
    
    # Calculate deviations from nominal values
    df['voltage_deviation'] = (df['voltage'] - NOMINAL_VOLTAGE) / NOMINAL_VOLTAGE
    df['frequency_deviation'] = (df['frequency'] - IDEAL_FREQUENCY) / IDEAL_FREQUENCY
    df['pf_deviation'] = df['powerFactor'] - IDEAL_POWER_FACTOR
    
    # DETAILED VOLTAGE ANOMALY CATEGORIES
    df['severe_voltage_dip'] = ((df['voltage'] < 195.0)).astype(int)
    df['moderate_voltage_dip'] = ((df['voltage'] >= 195.0) & (df['voltage'] < 207.0)).astype(int)
    df['mild_voltage_dip'] = ((df['voltage'] >= 207.0) & (df['voltage'] < 217.4)).astype(int)
    
    df['mild_surge'] = ((df['voltage'] > 242.6) & (df['voltage'] <= 248.0)).astype(int)
    df['moderate_surge'] = ((df['voltage'] > 248.0) & (df['voltage'] <= 253.0)).astype(int)
    df['severe_surge'] = ((df['voltage'] > 253.0)).astype(int)
    
    # DETAILED POWER FACTOR ANOMALY CATEGORIES
    df['severe_pf_issue'] = ((df['powerFactor'] < 0.5)).astype(int)
    df['moderate_pf_issue'] = ((df['powerFactor'] >= 0.5) & (df['powerFactor'] < 0.7)).astype(int)
    df['mild_pf_issue'] = ((df['powerFactor'] >= 0.7) & (df['powerFactor'] < 0.792)).astype(int)
    
    # FREQUENCY ANOMALIES
    df['freq_low'] = ((df['frequency'] < 59.2)).astype(int)
    df['freq_high'] = ((df['frequency'] > 60.8)).astype(int)
    
    # LOAD AND POWER ANOMALIES
    df['high_current'] = (df['current'] > 10.0).astype(int)
    df['very_high_current'] = (df['current'] > 20.0).astype(int)
    df['high_power'] = (df['power'] > 1000.0).astype(int)
    df['very_high_power'] = (df['power'] > 3000.0).astype(int)
    
    # Original basic flags (for compatibility)
    df['transient_flag'] = ((df['voltage'] < 207.0) & (df['frequency'] > 59.0) & 
                           (df['frequency'] < 61.0)).astype(int)
    df['surge_flag'] = ((df['voltage'] > 248.0) & (df['frequency'] > 59.0) & 
                       (df['frequency'] < 61.0)).astype(int)
    df['pf_issue_flag'] = (df['powerFactor'] < 0.75).astype(int)
    
    # Create ratios (useful for detecting unusual load patterns)
    df['power_voltage_ratio'] = df['power'] / (df['voltage'] + 0.1)
    df['current_voltage_ratio'] = df['current'] / (df['voltage'] + 0.1)
    
    return df

# 4. Reduce data while preserving meaningful patterns
def reduce_data(df, max_samples=20000):
    """Reduce dataset size while ensuring balanced representation of anomaly types."""
    print("\nReducing data...")
    original_len = len(df)
    print(f"Processing {original_len:,} rows to reduce to {max_samples:,} samples")
    
    if len(df) <= max_samples:
        print(f"Data already small enough ({len(df):,} rows)")
        return df
    
    # STEP 1: Identify rows by anomaly type
    transient_rows = df[df['transient_flag'] == 1]
    surge_rows = df[df['surge_flag'] == 1]
    pf_issue_rows = df[df['pf_issue_flag'] == 1]
    print(f"Found {len(transient_rows)} transient anomalies")
    print(f"Found {len(surge_rows)} surge anomalies")
    print(f"Found {len(pf_issue_rows)} power factor issues")
    
    # STEP 2: Calculate samples for each category
    # Allocate samples: 20% transients, 20% surges, 30% PF issues, 30% normal data
    # If a category has fewer rows than its allocation, take all available and redistribute
    transient_target = int(max_samples * 0.2)  # 20% for transients
    surge_target = int(max_samples * 0.2)      # 20% for surges
    pf_issue_target = int(max_samples * 0.3)   # 30% for PF issues
    normal_target = max_samples - transient_target - surge_target - pf_issue_target  # Remaining for normal
    
    # Adjust if we don't have enough rows of a certain type
    transient_sample_size = min(len(transient_rows), transient_target)
    remainder = transient_target - transient_sample_size
    
    surge_target += remainder // 3
    pf_issue_target += remainder // 3
    normal_target += remainder - (2 * (remainder // 3))
    
    surge_sample_size = min(len(surge_rows), surge_target)
    remainder = surge_target - surge_sample_size
    
    pf_issue_target += remainder // 2
    normal_target += remainder - (remainder // 2)
    
    pf_issue_sample_size = min(len(pf_issue_rows), pf_issue_target)
    normal_target += (pf_issue_target - pf_issue_sample_size)
    
    # STEP 3: Sample from each category
    sampled_transients = transient_rows.sample(transient_sample_size, random_state=42) if transient_sample_size > 0 else pd.DataFrame()
    sampled_surges = surge_rows.sample(surge_sample_size, random_state=42) if surge_sample_size > 0 else pd.DataFrame()
    sampled_pf_issues = pf_issue_rows.sample(pf_issue_sample_size, random_state=42) if pf_issue_sample_size > 0 else pd.DataFrame()
    
    # STEP 4: Get normal data (not flagged by any anomaly)
    all_anomalies = pd.concat([sampled_transients, sampled_surges, sampled_pf_issues])
    normal_data = df[~((df['transient_flag'] == 1) | (df['surge_flag'] == 1) | (df['pf_issue_flag'] == 1))]
    
    # STEP 5: Create stratified sample of normal data by location
    normal_sample = pd.DataFrame()
    if len(normal_data) > 0 and normal_target > 0:
        for location in normal_data['location'].unique():
            loc_data = normal_data[normal_data['location'] == location]
            loc_proportion = len(loc_data) / len(normal_data)
            loc_sample_size = max(50, int(normal_target * loc_proportion))  # At least 50 from each location
            
            if len(loc_data) > 0:
                # Create voltage bins for stratified sampling
                loc_data['voltage_bin'] = pd.qcut(loc_data['voltage'], 
                                                 q=min(10, len(loc_data)),
                                                 labels=False, 
                                                 duplicates='drop')
                
                loc_sample = pd.DataFrame()
                for bin_id in loc_data['voltage_bin'].unique():
                    bin_data = loc_data[loc_data['voltage_bin'] == bin_id]
                    bin_sample_size = max(1, int(loc_sample_size * len(bin_data) / len(loc_data)))
                    if len(bin_data) > 0:
                        bin_sample = bin_data.sample(min(bin_sample_size, len(bin_data)), random_state=42)
                        loc_sample = pd.concat([loc_sample, bin_sample])
                
                normal_sample = pd.concat([normal_sample, loc_sample])
    
    # If we still need more normal samples, add them randomly
    if len(normal_sample) < normal_target and len(normal_data) > 0:
        remaining_normal = normal_data[~normal_data.index.isin(normal_sample.index)]
        if len(remaining_normal) > 0:
            additional_samples = min(normal_target - len(normal_sample), len(remaining_normal))
            normal_sample = pd.concat([normal_sample, 
                                     remaining_normal.sample(additional_samples, random_state=42)])
    
    # STEP 6: Combine all samples
    reduced_df = pd.concat([sampled_transients, sampled_surges, sampled_pf_issues, normal_sample])
    
    # Ensure we don't exceed max_samples
    if len(reduced_df) > max_samples:
        reduced_df = reduced_df.sample(max_samples, random_state=42)
    
    print(f"Reduced data from {original_len:,} to {len(reduced_df):,} rows")
    print("\nAnomaly type distribution in sampled data:")
    print(f" - Transient anomalies: {sum(reduced_df['transient_flag'])} ({sum(reduced_df['transient_flag'])/len(reduced_df):.1%})")
    print(f" - Surge anomalies: {sum(reduced_df['surge_flag'])} ({sum(reduced_df['surge_flag'])/len(reduced_df):.1%})")
    print(f" - PF issues: {sum(reduced_df['pf_issue_flag'])} ({sum(reduced_df['pf_issue_flag'])/len(reduced_df):.1%})")
    print(f" - Normal data: {len(reduced_df) - sum(reduced_df['transient_flag'] | reduced_df['surge_flag'] | reduced_df['pf_issue_flag'])} ({(len(reduced_df) - sum(reduced_df['transient_flag'] | reduced_df['surge_flag'] | reduced_df['pf_issue_flag']))/len(reduced_df):.1%})")
    
    # Print location distribution
    loc_counts = reduced_df['location'].value_counts()
    print("\nSamples per location:")
    for loc, count in loc_counts.items():
        print(f" - {loc}: {count} rows")
        
    return reduced_df

# 5. Normalize/Scale data for clustering
def normalize_data(df, scaler_type='standard'):
    """Normalize numerical features for clustering."""
    print("\nNormalizing data...")
    
    # Select features for clustering
    feature_cols = [
        'voltage', 'current', 'frequency', 'power', 'powerFactor',
        'voltage_deviation', 'frequency_deviation', 'pf_deviation',
        'power_voltage_ratio', 'current_voltage_ratio'
    ]
    
    # Select appropriate scaler
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type.lower() == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit scaler and transform the data
    features = df[feature_cols].copy()
    scaled_features = scaler.fit_transform(features)
    
    # Create DataFrame with scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    
    # Add location and flag columns (not scaled)
    for col in ['location', 'transient_flag', 'surge_flag', 'pf_issue_flag']:
        if col in df.columns:
            scaled_df[col] = df[col].values
    
    print(f"Data normalized using {scaler_type} scaling")
    return scaled_df, feature_cols, scaler

# 6. Visualize the data
def visualize_data(df, scaled_df, feature_cols):
    """Create visualizations with better performance."""
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # 1. Distribution of key parameters
    print("  Creating parameter distributions plot...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['voltage', 'current', 'frequency', 'power', 'powerFactor']):
        plt.subplot(2, 3, i+1)
        sample_size = min(5000, len(df))
        sns.histplot(df[col].sample(sample_size), kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('plots/parameter_distributions.png')
    plt.close()
    
    # 2. Visualize patterns with PCA (2D projection)
    print("  Creating PCA projection plots...")
    pca = PCA(n_components=2)
    
    # Sample for PCA visualization
    sample_size = min(5000, len(scaled_df))
    sample_indices = np.random.choice(len(scaled_df), sample_size, replace=False)
    sampled_df = df.iloc[sample_indices]
    sampled_scaled_df = scaled_df.iloc[sample_indices]
    
    pca_result = pca.fit_transform(sampled_scaled_df[feature_cols])
    
    plt.figure(figsize=(12, 10))
    
    # Color by location - USING sampled_df
    print("  - Creating location PCA plot...")
    plt.subplot(2, 2, 1)
    for location in sampled_df['location'].unique():
        mask = sampled_df['location'] == location  # FIXED: using sampled_df
        if sum(mask) > 0:
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=location, alpha=0.6, s=10)
    plt.title('PCA Projection by Location')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    
    # Color by transient flag - USING sampled_df
    plt.subplot(2, 2, 2)
    for flag in [0, 1]:
        mask = sampled_df['transient_flag'] == flag  # FIXED: using sampled_df
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   label=f'Transient: {"Yes" if flag else "No"}', 
                   alpha=0.6, s=10)
    plt.title('PCA Projection by Transient Flag')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    
    # Color by surge flag - USING sampled_df
    plt.subplot(2, 2, 3)
    for flag in [0, 1]:
        mask = sampled_df['surge_flag'] == flag  # FIXED: using sampled_df
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   label=f'Surge: {"Yes" if flag else "No"}', 
                   alpha=0.6, s=10)
    plt.title('PCA Projection by Voltage Surge Flag')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    
    # Color by PF issue flag - USING sampled_df
    plt.subplot(2, 2, 4)
    for flag in [0, 1]:
        mask = sampled_df['pf_issue_flag'] == flag  # FIXED: using sampled_df
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   label=f'PF Issue: {"Yes" if flag else "No"}', 
                   alpha=0.6, s=10)
    plt.title('PCA Projection by Power Factor Issue Flag')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/pca_visualization.png')
    plt.close()  # Close the figure
    
    # 3. Correlation matrix of features
    print("  Creating correlation matrix...")
    plt.figure(figsize=(12, 10))
    corr_matrix = df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()  # Close the figure
    
    print("  Visualizations saved to 'plots' directory")
    return pca_result

def find_optimal_clusters(X, model_class, max_k=10, random_state=42, **model_params):
    """
    Find optimal number of clusters using silhouette method for any clustering model.
    
    Parameters:
    - X: Feature matrix
    - model_class: The clustering class to use (e.g., KMeans, AgglomerativeClustering)
    - max_k: Maximum number of clusters to try
    - random_state: Random seed for reproducibility
    - **model_params: Additional parameters to pass to the model
    
    Returns:
    - optimal_k: Optimal number of clusters
    - best_score: Best silhouette score achieved
    - scores: List of silhouette scores for each k
    """
    print(f"\nFinding optimal clusters for {model_class.__name__}...")
    best_k = 2  # minimum 2 clusters
    best_score = -1
    scores = []
    
    for k in range(2, max_k+1):
        # Create model instance with current k
        try:
            if model_class == KMeans:
                model = model_class(n_clusters=k, random_state=random_state, **model_params)
            elif model_class == GaussianMixture:
                model = model_class(n_components=k, random_state=random_state, **model_params)
            elif model_class == SpectralClustering:
                model = model_class(n_clusters=k, random_state=random_state, **model_params)
            elif model_class == AgglomerativeClustering:
                model = model_class(n_clusters=k, **model_params)
            else:
                print(f"  Unsupported model class: {model_class.__name__}")
                return 2, -1, []
            
            # Fit model and get cluster labels
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                labels = model.predict(X)
            
            # Calculate silhouette score
            score = silhouette_score(X, labels)
            scores.append(score)
            print(f"  k={k}: silhouette={score:.4f}")
            
            # Update best score and k if better
            if score > best_score:
                best_score = score
                best_k = k
                
        except Exception as e:
            print(f"  Error with k={k}: {str(e)}")
            scores.append(float('nan'))
    
    print(f"  Best k for {model_class.__name__}: {best_k} (score={best_score:.4f})")
    return best_k, best_score, scores

# 7. Perform clustering and evaluate
def perform_clustering(scaled_df, feature_cols):
    """Perform clustering using multiple methods and evaluate results."""
    print("\nPerforming clustering analysis...")
    
    # Prepare data for clustering
    X = scaled_df[feature_cols].values
    total_samples = X.shape[0]
    print(f"Clustering {total_samples:,} data points across {len(feature_cols)} features")
    
    # Define clustering model classes to try
    model_classes = {
        'KMeans': KMeans,
        'SpectralClustering': SpectralClustering,
        'AgglomerativeClustering': AgglomerativeClustering,
        'WardHierarchical': AgglomerativeClustering,
        'GaussianMixture': GaussianMixture
    }
    
    n_clusters, _, _ = find_optimal_clusters(X, model_classes['KMeans'])
    
    # # Find optimal k for each model type
    # optimal_ks = {}
    # for name, model_class in model_classes.items():
    #     if name == 'WardHierarchical':
    #         optimal_ks[name], _, _ = find_optimal_clusters(X, model_class, linkage='ward')
    #     elif name == 'SpectralClustering':
    #         optimal_ks[name], _, _ = find_optimal_clusters(X, model_class, affinity='nearest_neighbors')
    #     else:
    #         optimal_ks[name], _, _ = find_optimal_clusters(X, model_class)
    
    # # Create models with their optimal k
    # clustering_models = {
    #     'KMeans': KMeans(n_clusters=optimal_ks['KMeans'], random_state=42, verbose=1),
    #     'SpectralClustering': SpectralClustering(n_clusters=optimal_ks['SpectralClustering'], random_state=42, affinity='nearest_neighbors'),
    #     'AgglomerativeClustering': AgglomerativeClustering(n_clusters=optimal_ks['AgglomerativeClustering']),
    #     'WardHierarchical': AgglomerativeClustering(n_clusters=optimal_ks['WardHierarchical'], linkage='ward'),
    #     'GaussianMixture': GaussianMixture(n_components=optimal_ks['GaussianMixture'], random_state=42, verbose=1)
    # }
      # Create models with their optimal k
    clustering_models = {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42, verbose=1),
    }
    
    # Dictionary to store results
    results = {}
    
    # Perform clustering and calculate metrics
    for name, model in clustering_models.items():
        start_time = time.time()
        print(f"\nRunning {name}... [Started at {datetime.now().strftime('%H:%M:%S')}]")
        
        # Fit the model and get labels
        if hasattr(model, 'fit_predict'):
            print(f"  - Fitting and predicting with {name}...")
            labels = model.fit_predict(X)
        else:
            print(f"  - Fitting {name}...")
            model.fit(X)
            print(f"  - Predicting with {name}...")
            labels = model.predict(X)
        
        # Calculate evaluation metrics
        metrics = {}
        print("  - Calculating evaluation metrics...")
        
        try:
            print("    - Computing Silhouette Score...")
            metrics['silhouette'] = silhouette_score(X, labels)
        except Exception as e:
            print(f"    - Silhouette Score failed: {str(e)}")
            metrics['silhouette'] = float('nan')
            
        try:
            print("    - Computing Davies-Bouldin Index...")
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        except Exception as e:
            print(f"    - Davies-Bouldin Index failed: {str(e)}")
            metrics['davies_bouldin'] = float('nan')
            
        try:
            print("    - Computing Calinski-Harabasz Index...")
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        except Exception as e:
            print(f"    - Calinski-Harabasz Index failed: {str(e)}")
            metrics['calinski_harabasz'] = float('nan')
        
        # Store results
        results[name] = {
            'labels': labels,
            'metrics': metrics
        }
        
        # Calculate and display time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_str = f"{int(elapsed_time // 60)} mins {int(elapsed_time % 60)} secs"
        
        print(f"  {name} completed in {time_str}")
        print(f"  Silhouette: {metrics['silhouette']:.4f}")
        print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
        print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.4f}")
    
    # Create dataframe of metrics for comparison
    metrics_df = pd.DataFrame({
        name: {
            'Silhouette Score': results[name]['metrics']['silhouette'],
            'Davies-Bouldin Index': results[name]['metrics']['davies_bouldin'],
            'Calinski-Harabasz Index': results[name]['metrics']['calinski_harabasz']
        } for name in results
    })
    
    # Find best model based on silhouette score
    best_model = max(results.keys(), key=lambda x: results[x]['metrics']['silhouette'] 
                     if not np.isnan(results[x]['metrics']['silhouette']) else -np.inf)
    
    # Add best cluster labels to df
    scaled_df['cluster'] = results[best_model]['labels']
    
    print(f"\nBest clustering model: {best_model}")
    print(metrics_df[best_model])
    
    return results, metrics_df, best_model

# 8. Analyze clusters
def analyze_clusters(df, scaled_df, results, best_model):
    """Analyze the clusters to determine what they represent."""
    print("\nAnalyzing clusters...")
    
    cluster_labels = results[best_model]['labels']
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Count samples per cluster
    cluster_counts = df_with_clusters['cluster'].value_counts().sort_index()
    print("Samples per cluster:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} samples ({count/len(df_with_clusters):.1%})")
    
    # Calculate cluster profiles - INCLUDE ALL DETAILED ANOMALY CATEGORIES
    cluster_profiles = df_with_clusters.groupby('cluster')[
        ['voltage', 'current', 'frequency', 'power', 'powerFactor', 
        'transient_flag', 'surge_flag', 'pf_issue_flag',
        'severe_voltage_dip', 'moderate_voltage_dip', 'mild_voltage_dip',
        'mild_surge', 'moderate_surge', 'severe_surge', 
        'severe_pf_issue', 'moderate_pf_issue', 'mild_pf_issue',
        'freq_low', 'freq_high',
        'high_current', 'very_high_current', 'high_power', 'very_high_power',
        'voltage_deviation', 'frequency_deviation', 'pf_deviation',  # Add these three lines
        'power_voltage_ratio', 'current_voltage_ratio'              # Add derived ratios too
        ]
    ].mean()
    
    print("\nCluster profiles:")
    print(cluster_profiles)
    
    # Create 2D visualization of clusters with PCA
    pca = PCA(n_components=2)
    X = scaled_df.drop(columns=['location', 'transient_flag', 'surge_flag', 'pf_issue_flag', 'cluster']).values
    pca_result = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for cluster in np.unique(cluster_labels):
        plt.scatter(
            pca_result[cluster_labels == cluster, 0],
            pca_result[cluster_labels == cluster, 1],
            label=f'Cluster {cluster}', alpha=0.7, s=15
        )
    
    plt.title(f'Clustering Results with {best_model}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/clustering_{best_model}.png')
    
    # Map clusters to anomaly types with detailed categories
    anomaly_mapping = {}
    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]
        
        # Start with most severe conditions and work down
        if profile['current'] < 0.01 and profile['power'] < 1.0:
            anomaly_type = "No Load (Idle Operation)"
        # Light load with acceptable PF
        elif profile['current'] < 2.0 and profile['power'] < 400.0 and profile['powerFactor'] >= 0.75:
            anomaly_type = "Light Load (Normal Operation)"
        # Then continue with your existing conditions
        elif profile['severe_voltage_dip'] > 0.3:
            anomaly_type = "Severe Voltage Dip"
        elif profile['severe_surge'] > 0.3:
            anomaly_type = "Severe Voltage Surge"
        elif profile['very_high_power'] > 0.5 and profile['powerFactor'] > 0.9:
            anomaly_type = "Very High Load (Good PF)"
        elif profile['very_high_power'] > 0.5 and profile['powerFactor'] < 0.75:
            anomaly_type = "Very High Load with PF Issues"
        elif profile['high_current'] > 0.5 and profile['moderate_pf_issue'] > 0.5:
            anomaly_type = "High Current with PF Issues"
        elif profile['moderate_voltage_dip'] > 0.3:
            anomaly_type = "Moderate Voltage Dip"
        elif profile['moderate_surge'] > 0.3:
            anomaly_type = "Moderate Voltage Surge"
        elif profile['severe_pf_issue'] > 0.3:
            anomaly_type = "Severe Power Factor Issue"
        elif profile['moderate_pf_issue'] > 0.5:
            anomaly_type = "Moderate Power Factor Issue"
        elif profile['mild_pf_issue'] > 0.5:
            anomaly_type = "Mild Power Factor Issue"
        elif profile['freq_low'] > 0.3:
            anomaly_type = "Low Frequency"
        elif profile['freq_high'] > 0.3:
            anomaly_type = "High Frequency"
        elif (profile['power'] > 1000 and profile['current'] > 5):
            anomaly_type = "High Load Operation"
        elif profile['pf_deviation'] > 0.1:
            anomaly_type = "Minor Power Factor Deviation"
        else:
            anomaly_type = "Normal Operation"
        
        anomaly_mapping[cluster] = anomaly_type
    
    print("\nCluster to anomaly type mapping:")
    for cluster, anomaly_type in anomaly_mapping.items():
        print(f"  Cluster {cluster} → {anomaly_type}")
    
    # Save results for further analysis
    df_with_clusters['anomaly_type'] = df_with_clusters['cluster'].map(anomaly_mapping)
    df_with_clusters.to_csv('clustered_anomalies.csv', index=False)
    
    return df_with_clusters, anomaly_mapping

# Main execution
if __name__ == "__main__":
    print("=== Power Quality Anomaly Analysis ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    overall_start = time.time()
    
    # 1. Load data
    step_start = time.time()
    print("\n[Step 1/8] Loading anomalous data...")
    df = load_anomalous_data()
    print(f"Step 1 completed in {time.time() - step_start:.1f} seconds")
    
    # 2. Preprocess data
    step_start = time.time()
    print("\n[Step 2/8] Preprocessing data...")
    df = preprocess_data(df)
    print(f"Step 2 completed in {time.time() - step_start:.1f} seconds")
    
    # 3. Engineer features
    step_start = time.time()
    print("\n[Step 3/8] Engineering features...")
    df = engineer_features(df)
    print(f"Step 3 completed in {time.time() - step_start:.1f} seconds")
    
    # 4. Reduce data while preserving patterns
    step_start = time.time()
    print("\n[Step 4/8] Reducing data...")
    df_reduced = reduce_data(df, max_samples=15000)  # Changed to 20,000 as per your suggestion
    print(f"Step 4 completed in {time.time() - step_start:.1f} seconds")
    
    # 5. Normalize data
    step_start = time.time()
    print("\n[Step 5/8] Normalizing data...")
    scaled_df, feature_cols, scaler = normalize_data(df_reduced)
    print(f"Step 5 completed in {time.time() - step_start:.1f} seconds")
    
    # # 6. Visualize data
    # step_start = time.time()
    # print("\n[Step 6/8] Creating visualizations...")
    # pca_result = visualize_data(df_reduced, scaled_df, feature_cols)
    # print(f"Step 6 completed in {time.time() - step_start:.1f} seconds")
    
    # 7. Perform clustering and evaluate
    step_start = time.time()
    print("\n[Step 7/8] Performing clustering analysis...")
    results, metrics_df, best_model = perform_clustering(scaled_df, feature_cols)
    print(f"Step 7 completed in {time.time() - step_start:.1f} seconds")
    
    # 8. Analyze clusters
    step_start = time.time()
    print("\n[Step 8/8] Analyzing clusters...")
    df_with_clusters, anomaly_mapping = analyze_clusters(df_reduced, scaled_df, results, best_model)
    print(f"Step 8 completed in {time.time() - step_start:.1f} seconds")
    
    overall_time = time.time() - overall_start
    hours = int(overall_time // 3600)
    minutes = int((overall_time % 3600) // 60)
    seconds = int(overall_time % 60)
    
    print("\n=== Analysis Complete ===")
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
    print(f"Results saved to 'clustered_anomalies.csv'")
    print(f"Visualizations saved to 'plots/' directory")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")