import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def debug_data_loading(file_path):
    """Debug data loading issues"""
    print("=== DATA LOADING DEBUG ===")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return None
    
    try:
        # Try loading the data
        data = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {len(data.columns)}")
        print(f"  First few column names: {list(data.columns[:10])}")
        return data
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

def analyze_variables(data, x_variables):
    """Analyze which variables are available"""
    print("\n=== VARIABLE ANALYSIS ===")
    
    available_vars = [var for var in x_variables if var in data.columns]
    missing_vars = [var for var in x_variables if var not in data.columns]
    
    print(f"Total variables requested: {len(x_variables)}")
    print(f"Available variables: {len(available_vars)}")
    print(f"Missing variables: {len(missing_vars)}")
    
    if missing_vars:
        print(f"\nFirst 20 missing variables:")
        for i, var in enumerate(missing_vars[:20]):
            print(f"  {i+1}. {var}")
    
    return available_vars, missing_vars

def analyze_missing_data(X):
    """Analyze missing data patterns"""
    print("\n=== MISSING DATA ANALYSIS ===")
    print(f"Dataset shape: {X.shape}")
    
    # Missing values per column
    missing_per_col = X.isnull().sum()
    missing_pct_per_col = (missing_per_col / len(X)) * 100
    
    print(f"Total missing values: {missing_per_col.sum()}")
    print(f"Columns with missing data: {(missing_per_col > 0).sum()}")
    
    # Show worst columns
    worst_missing = missing_pct_per_col.sort_values(ascending=False).head(10)
    print(f"\nColumns with most missing data (%):")
    for col, pct in worst_missing.items():
        print(f"  {col}: {pct:.1f}%")
    
    return missing_per_col, missing_pct_per_col

def perform_pca_analysis():
    # Update this path to your actual file location
    file_path = "x"
    
    # Define your variables (keeping your original list)
    x_variables = [
        'nwspol', 'netusoft', 'happy', 'sclmeet', 'inprdsc', 'sclact', 'crmvct', 'aesfdrk', 'health', 'hlthhmp',
        'atchctr', 'atcherp', 'rlgblg', 'rlgdnm', 'rlgdnbat', 'rlgdnacy', 'rlgdnafi', 'rlgdnade', 'rlgdnagr',
        'rlgdnhu', 'rlgdnais', 'rlgdnie', 'rlgdnlt', 'rlgdnanl', 'rlgdnno', 'rlgdnapl', 'rlgdnapt', 'rlgdnrs',
        'rlgdnask', 'rlgdnase', 'rlgdnach', 'rlgdngb', 'rlgblge', 'rlgdnme', 'rlgdebat', 'rlgdeacy', 'rlgdeafi',
        'rlgdeade', 'rlgdeagr', 'rlgdehu', 'rlgdeais', 'rlgdeie', 'rlgdelt', 'rlgdeanl', 'rlgdeno', 'rlgdeapl',
        'rlgdeapt', 'rlgders', 'rlgdeask', 'rlgdease', 'rlgdeach', 'rlgdegb', 'rlgdgr', 'rlgatnd', 'pray',
        'dscrgrp', 'dscrrce', 'dscrntn', 'dscrrlg', 'dscrlng', 'dscretn', 'dscrage', 'dscrgnd', 'dscrsex',
        'dscrdsb', 'dscroth', 'dscrdk', 'dscrref', 'dscrnap', 'dscrna', 'ctzcntr', 'brncntr', 'cntbrthd',
        'livecnta', 'lnghom1', 'lnghom2', 'feethngr', 'facntr', 'fbrncntc', 'mocntr', 'mbrncntc', 'ccnthum',
        'ccrdprs', 'wrclmch', 'admrclc', 'testjc34', 'testjc35', 'testjc36', 'testjc37', 'testjc38', 'testjc39',
        'testjc40', 'testjc41', 'testjc42', 'vteurmmb', 'vteubcmb', 'ctrlife', 'etfruit', 'eatveg', 'dosprt',
        'cgtsmok', 'alcfreq', 'alcwkdy', 'alcwknd', 'icgndra', 'alcbnge', 'height', 'weighta', 'dshltgp',
        'dshltms', 'dshltnt', 'dshltref', 'dshltdk', 'dshltna', 'medtrun', 'medtrnp', 'medtrnt', 'medtroc',
        'medtrnl', 'medtrwl', 'medtrnaa', 'medtroth', 'medtrnap', 'medtrref', 'medtrdk', 'medtrna', 'medtrnu',
        'hlpfmly', 'hlpfmhr', 'trhltacu', 'trhltacp', 'trhltcm', 'trhltch', 'trhltos', 'trhltho', 'trhltht',
        'trhlthy', 'trhltmt', 'trhltpt', 'trhltre', 'trhltsh', 'trhltnt', 'trhltref', 'trhltdk', 'trhltna',
        'fltdpr', 'flteeff', 'slprl', 'wrhpp', 'fltlnl', 'enjlf', 'fltsd', 'cldgng', 'hltprhc', 'hltprhb',
        'hltprbp', 'hltpral', 'hltprbn', 'hltprpa', 'hltprpf', 'hltprsd', 'hltprsc', 'hltprsh', 'hltprdi',
        'hltprnt', 'hltprref', 'hltprdk', 'hltprna', 'hltphhc', 'hltphhb', 'hltphbp', 'hltphal', 'hltphbn',
        'hltphpa', 'hltphpf', 'hltphsd', 'hltphsc', 'hltphsh', 'hltphdi', 'hltphnt', 'hltphnap', 'hltphref',
        'hltphdk', 'hltphna', 'hltprca', 'cancfre', 'cnfpplh', 'fnsdfml', 'jbexpvi', 'jbexpti', 'jbexpml',
        'jbexpmc', 'jbexpnt', 'jbexpnap', 'jbexpref', 'jbexpdk', 'jbexpna', 'jbexevl', 'jbexevh', 'jbexevc',
        'jbexera', 'jbexecp', 'jbexebs', 'jbexent', 'jbexenap', 'jbexeref', 'jbexedk', 'jbexena', 'netustm',
        'ppltrst', 'pplfair', 'pplhlp', 'rshpsts', 'rshpsgb', 'lvgptnea', 'dvrcdeva', 'marsts', 'marstgb',
        'maritalb', 'chldhhe', 'domicil', 'paccmoro', 'paccdwlr', 'pacclift', 'paccnbsh', 'paccocrw', 'paccxhoc',
        'paccnois', 'paccinro', 'paccnt', 'paccref', 'paccdk', 'paccna', 'edulvlb', 'eisced', 'edlveat',
        'edlvebe', 'edlvehr', 'edlvgcy', 'edlvdfi', 'edlvdfr', 'edudde1', 'educde2', 'edlvegr', 'edlvdahu',
        'edlvdis', 'edlvdie', 'edlvfit', 'edlvdlt', 'edlvenl', 'edlveno', 'edlvipl', 'edlvept', 'edlvdrs',
        'edlvdsk', 'edlvesi', 'edlvies', 'edlvdse', 'edlvdch', 'educgb1', 'edubgb2', 'edagegb', 'eduyrs',
        'pdwrk', 'edctn', 'uempla', 'uempli', 'dsbld', 'rtrd', 'cmsrv', 'hswrk', 'dngoth', 'dngref', 'dngdk',
        'dngna', 'mainact', 'mnactic', 'crpdwk', 'pdjobev', 'pdjobyr', 'emplrel', 'emplno', 'wrkctra', 'estsz',
        'jbspv', 'njbspv', 'wkdcorga', 'iorgact', 'wkhct', 'wkhtot', 'nacer2', 'tporgwk', 'isco08', 'wrkac6m',
        'uemp3m', 'uemp12m', 'uemp5yr', 'mbtru', 'hincsrca', 'hinctnta', 'hincfel', 'edulvlpb', 'eiscedp',
        'edlvpfat', 'edlvpebe', 'edlvpehr', 'edlvpgcy', 'edlvpdfi', 'edlvpdfr', 'edupdde1', 'edupcde2',
        'edlvpegr', 'edlvpdahu', 'edlvpdis', 'edlvpdie', 'edlvpfit', 'edlvpdlt', 'edlvpenl', 'edlvpeno',
        'edlvphpl', 'edlvpept', 'edlvpdrs', 'edlvpdsk', 'edlvpesi', 'edlvphes', 'edlvpdse', 'edlvpdch',
        'edupcgb1', 'edupbgb2', 'edagepgb', 'pdwrkp', 'edctnp', 'uemplap', 'uemplip', 'dsbldp', 'rtrdp',
        'cmsrvp', 'hswrkp', 'dngothp', 'dngdkp', 'dngnapp', 'dngrefp', 'dngnap', 'mnactp', 'crpdwkp', 'isco08p',
        'emprelp', 'wkhtotp', 'edulvlfb', 'eiscedf', 'edlvfeat', 'edlvfebe', 'edlvfehr', 'edlvfgcy', 'edlvfdfi',
        'edlvfdfr', 'edufcde1', 'edufbde2', 'edlvfegr', 'edlvfdahu', 'edlvfdis', 'edlvfdie', 'edlvffit',
        'edlvfdlt', 'edlvfenl', 'edlvfeno', 'edlvfgpl', 'edlvfept', 'edlvfdrs', 'edlvfdsk', 'edlvfesi',
        'edlvfges', 'edlvfdse', 'edlvfdch', 'edufcgb1', 'edufbgb2', 'edagefgb', 'emprf14', 'occf14b',
        'edulvlmb', 'eiscedm', 'edlvmeat', 'edlvmebe', 'edlvmehr', 'edlvmgcy', 'edlvmdfi', 'edlvmdfr',
        'edumcde1', 'edumbde2', 'edlvmegr', 'edlvmdahu', 'edlvmdis', 'edlvmdie', 'edlvmfit', 'edlvmdlt',
        'edlvmenl', 'edlvmeno', 'edlvmgpl', 'edlvmept', 'edlvmdrs', 'edlvmdsk', 'edlvmesi', 'edlvmges',
        'edlvmdse', 'edlvmdch', 'edumcgb1', 'edumbgb2', 'edagemgb', 'emprm14', 'occm14b', 'atncrse',
        'anctrya1', 'anctrya2', 'regunit', 'region'
    ]
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Step 1: Debug data loading
    data = debug_data_loading(file_path)
    if data is None:
        return None, None, None
    
    # Step 2: Analyze available variables
    available_vars, missing_vars = analyze_variables(data, x_variables)
    
    if len(available_vars) == 0:
        print("ERROR: No variables from your list are found in the dataset!")
        print("This suggests column name mismatch. Check the actual column names in your dataset.")
        return None, None, None
    
    # Step 3: Create subset with available variables
    X = data[available_vars].copy()
    
    # Step 4: Analyze missing data
    missing_per_col, missing_pct_per_col = analyze_missing_data(X)
    
    # Step 5: Data cleaning with proper data type handling
    print("\n=== DATA CLEANING ===")
    
    # Remove columns with more than 70% missing data (less strict)
    missing_threshold = 0.7
    cols_to_keep = X.columns[missing_pct_per_col < (missing_threshold * 100)]
    X_cleaned = X[cols_to_keep]
    print(f"After removing columns with >{missing_threshold*100}% missing: {X_cleaned.shape}")
    
    if X_cleaned.shape[1] == 0:
        print("ERROR: No columns remain after cleaning!")
        return None, None, None
    
    # Step 5a: Identify and handle data types
    print("\n=== DATA TYPE ANALYSIS ===")
    
    # Separate numeric and non-numeric columns
    numeric_cols = []
    non_numeric_cols = []
    
    for col in X_cleaned.columns:
        # Try to convert to numeric
        try:
            pd.to_numeric(X_cleaned[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            # Check if it's categorical with few unique values
            unique_vals = X_cleaned[col].dropna().unique()
            if len(unique_vals) <= 20:  # Categorical with few categories
                # Try to convert to numeric codes
                try:
                    X_cleaned[col] = pd.Categorical(X_cleaned[col]).codes
                    X_cleaned[col] = X_cleaned[col].replace(-1, np.nan)  # Replace missing category codes
                    numeric_cols.append(col)
                    print(f"  Converted categorical '{col}' to numeric codes ({len(unique_vals)} categories)")
                except:
                    non_numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Non-numeric columns: {len(non_numeric_cols)}")
    
    if non_numeric_cols:
        print(f"Removing non-numeric columns: {non_numeric_cols[:10]}...")  # Show first 10
        X_cleaned = X_cleaned[numeric_cols]
    
    print(f"Final shape after data type cleaning: {X_cleaned.shape}")
    
    if X_cleaned.shape[1] == 0:
        print("ERROR: No numeric columns remain!")
        return None, None, None
    
    # Handle remaining missing values with multiple strategies
    print("\nTrying different missing value strategies...")
    
    # Strategy 1: Drop rows with any missing values
    X_dropna = X_cleaned.dropna()
    print(f"Strategy 1 (dropna): {X_dropna.shape[0]} rows remaining")
    
    # Strategy 2: Fill missing values with median (now safe for numeric data)
    try:
        X_filled = X_cleaned.fillna(X_cleaned.median())
        print(f"Strategy 2 (median fill): {X_filled.shape[0]} rows remaining")
    except Exception as e:
        print(f"Median fill failed: {e}")
        X_filled = X_cleaned.copy()
        X_filled = X_filled.fillna(0)  # Fallback to zero fill
        print(f"Strategy 2 (zero fill): {X_filled.shape[0]} rows remaining")
    
    # Choose the best strategy
    if X_dropna.shape[0] >= 100:  # Need at least 100 observations
        X_final = X_dropna
        strategy_used = "dropna"
    elif X_filled.shape[0] >= 100:
        X_final = X_filled
        strategy_used = "median_fill"
    else:
        print("ERROR: Not enough data remaining after cleaning!")
        return None, None, None
    
    print(f"Using strategy: {strategy_used}")
    print(f"Final dataset shape: {X_final.shape}")
    
    # Step 6: Check for constant/near-constant columns
    print("\n=== CHECKING FOR PROBLEMATIC VARIABLES ===")
    
    # Remove constant columns
    constant_cols = []
    for col in X_final.columns:
        if X_final[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        X_final = X_final.drop(columns=constant_cols)
    
    # Remove columns with very low variance (optional)
    low_var_cols = []
    for col in X_final.columns:
        if X_final[col].std() < 1e-10:  # Very low standard deviation
            low_var_cols.append(col)
    
    if low_var_cols:
        print(f"Removing {len(low_var_cols)} low-variance columns: {low_var_cols}")
        X_final = X_final.drop(columns=low_var_cols)
    
    # Check final data types and ensure all are numeric
    print(f"Final shape: {X_final.shape}")
    print(f"Data types: {X_final.dtypes.value_counts().to_dict()}")
    
    # Force convert any remaining non-numeric columns
    for col in X_final.columns:
        if not pd.api.types.is_numeric_dtype(X_final[col]):
            print(f"Warning: Converting {col} to numeric")
            X_final[col] = pd.to_numeric(X_final[col], errors='coerce')
    
    # Remove any columns that became all NaN
    X_final = X_final.dropna(axis=1, how='all')
    
    if X_final.shape[1] < 2:
        print("ERROR: Need at least 2 variables for PCA!")
        return None, None, None
    
    # Step 7: Apply PCA
    print("\n=== APPLYING PCA ===")
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_final)
        print("✓ Data standardized successfully")
        
        # Apply PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        print("✓ PCA applied successfully")
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"\nExplained Variance (first 10 components):")
        for i in range(min(10, len(explained_variance))):
            print(f"  PC{i+1}: {explained_variance[i]:.4f} ({cumulative_variance[i]:.4f} cumulative)")
        
        # Visualizations
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # 1. Scree plot
        n_plot = min(20, len(explained_variance))
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, n_plot + 1), explained_variance[:n_plot], alpha=0.6, color='skyblue', label='Individual')
        plt.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 'ro-', label='Cumulative')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        plt.title('PCA Explained Variance (Scree Plot)')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/pca_scree_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Biplot for first two components
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30, color='blue')
        plt.title('PCA Biplot (First Two Components)')
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/pca_biplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary results
        n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        
        results = {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'n_components_80pct': n_components_80,
            'n_variables_used': X_final.shape[1],
            'n_observations': X_final.shape[0],
            'variables_available': len(available_vars),
            'variables_missing': len(missing_vars),
            'cleaning_strategy': strategy_used,
            'model_type': 'Principal Component Analysis'
        }
        
        print("\n=== PCA ANALYSIS COMPLETED SUCCESSFULLY! ===")
        print(f"Final results:")
        print(f"  Variables used: {results['n_variables_used']}")
        print(f"  Observations: {results['n_observations']}")
        print(f"  Components for 80% variance: {results['n_components_80pct']}")
        print(f"  Data cleaning strategy: {results['cleaning_strategy']}")
        
        return pca, X_pca, results
        
    except Exception as e:
        print(f"ERROR during PCA: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Run the analysis
if __name__ == "__main__":
    pca_model, pca_data, pca_results = perform_pca_analysis()
