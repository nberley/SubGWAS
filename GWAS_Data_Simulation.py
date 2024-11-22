# %% [markdown]
# # GWAS Data Simulation for Subtype Analysis
# 
# This notebook generates simulated GWAS summary statistics with hidden subtype structure.

# %%
# Import required packages
import numpy as np
import pandas as pd
from scipy.stats import norm

# %%
def simulate_gwas_data(n_variants=100000, n_causal=1000, n_subtypes=3, seed=42):
    np.random.seed(seed)
    
    # Create base dataframe
    gwas_data = pd.DataFrame({
        'SNP': [f'rs{i+1}' for i in range(n_variants)],
        'CHR': np.random.randint(1, 23, n_variants),
        'BP': np.random.randint(1, 250000000, n_variants),
        'A1': np.random.choice(['A', 'C', 'G', 'T'], n_variants),
        'A2': np.random.choice(['A', 'C', 'G', 'T'], n_variants),
        'MAF': np.random.uniform(0.01, 0.5, n_variants)
    })
    
    # Sort by CHR and BP
    gwas_data = gwas_data.sort_values(['CHR', 'BP']).reset_index(drop=True)
    
    # Simulate causal variants for each subtype
    causal_variants = np.random.choice(n_variants, n_causal, replace=False)
    
    # Initialize effect sizes and p-values
    betas = np.zeros((n_variants, n_subtypes))
    ses = np.ones((n_variants, n_subtypes)) * 0.02
    
    # Generate subtype-specific effects
    for subtype in range(n_subtypes):
        # Select subtype-specific causal variants
        subtype_causal = np.random.choice(
            causal_variants, 
            size=int(n_causal/2), 
            replace=False
        )
        
        # Generate effect sizes for causal variants
        effect_sizes = np.random.normal(0, 0.2, len(subtype_causal))
        betas[subtype_causal, subtype] = effect_sizes
        
        # Add some shared effects across subtypes
        if subtype > 0:
            shared_variants = np.random.choice(
                causal_variants, 
                size=int(n_causal/4), 
                replace=False
            )
            betas[shared_variants, subtype] = betas[shared_variants, 0]
    
    # Calculate z-scores and p-values
    z_scores = betas / ses
    
    # Calculate p-values with minimum threshold
    def safe_p_values(z_scores):
        raw_p = 2 * (1 - norm.cdf(abs(z_scores)))
        return np.maximum(raw_p, 1e-300)
    
    p_values = safe_p_values(z_scores)
    
    # Combine effects across subtypes (weighted average)
    weights = np.random.dirichlet(np.ones(n_subtypes))
    combined_betas = np.average(betas, axis=1, weights=weights)
    combined_ses = np.sqrt(np.average(ses**2, axis=1, weights=weights))
    combined_p = safe_p_values(combined_betas/combined_ses)
    
    # Add to dataframe
    gwas_data['BETA'] = combined_betas
    gwas_data['SE'] = combined_ses
    gwas_data['P'] = combined_p
    gwas_data['Z'] = combined_betas / combined_ses
    
    # Add subtype-specific effects
    for i in range(n_subtypes):
        gwas_data[f'BETA_subtype_{i+1}'] = betas[:, i]
        gwas_data[f'P_subtype_{i+1}'] = p_values[:, i]
    
    # Add LD blocks
    gwas_data['LD_block'] = np.repeat(
        range(int(n_variants/100)), 
        100
    )[:n_variants]
    
    return gwas_data

# %%
# Generate and save the data
gwas_data = simulate_gwas_data()
gwas_data.to_csv("simulated_gwas.csv", index=False)

print("Data preview:")
print(gwas_data.head())

print("Summary statistics:")
print(f"Total variants: {len(gwas_data)}")
print(f"Significant variants (p < 5e-8): {sum(gwas_data['P'] < 5e-8)}")


