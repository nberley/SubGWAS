# GWAS Clustering Analysis Tool

This tool performs unsupervised clustering analysis on Genome-Wide Association Study (GWAS) summary statistics to identify potential disease subtypes. It combines deep learning-based dimensionality reduction with clustering techniques to discover patterns in genetic data.

## Overview

The tool implements a two-step approach:
1. Dimensionality reduction using a deep autoencoder
2. Clustering analysis on the reduced representations

This approach helps identify distinct patterns in GWAS data that might correspond to different disease subtypes or genetic architectures.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn

Install dependencies using:
```bash
pip install torch numpy pandas scikit-learn
```

## Input Data Format

The script expects a CSV file containing GWAS summary statistics with the following columns:
- `snp`: SNP identifier
- `beta`: Effect size
- `se`: Standard error
- `p_value`: Association p-value
- `maf`: Minor allele frequency

Additional columns will be preserved but not used in the analysis.

## Usage

```python
from gwas_clustering import GWASClusterAnalysis

# Initialize the analyzer
analyzer = GWASClusterAnalysis(
    encoding_dim=32,  # Dimension of the encoded representation
    n_clusters_range=(2, 10)  # Range of cluster numbers to try
)

# Run the complete analysis
main("path_to_your_gwas_summary_stats.csv")
```

## Key Components

### 1. GWASAutoencoder

A deep autoencoder network that performs dimensionality reduction:
- Input layer: Matches the number of features in your GWAS data
- Hidden layers: 256 → 128 → encoding_dim
- Symmetric decoder architecture

### 2. GWASClusterAnalysis

Main class that orchestrates the analysis:

#### Preprocessing
- Handles missing values
- Log-transforms p-values
- Standardizes features

#### Autoencoder Training
- Reduces high-dimensional GWAS data to a lower-dimensional representation
- Uses MSE loss and Adam optimizer
- Configurable number of training epochs

#### Clustering
- Automatically determines optimal number of clusters
- Uses modified k-means algorithm
- Evaluates clustering quality using silhouette score

#### Cluster Analysis
- Computes cluster-specific statistics
- Identifies characteristic SNPs for each cluster
- Generates summary reports

## Output

The tool provides:
1. Optimal number of clusters
2. Silhouette score indicating clustering quality
3. For each cluster:
   - Size (number of SNPs)
   - Mean effect size (beta)
   - Mean minor allele frequency
   - Top associated SNPs

## Example Output

```
Optimal number of clusters: 4
Silhouette score: 0.723

CLUSTER_0:
Size: 1250
Mean beta: 0.156
Mean MAF: 0.234
Top SNPs: rs123, rs456, rs789, rs012, rs345
...
```

## Customization

### Modifying Feature Selection

To use different features for clustering, modify the `features` list in the `preprocess_data` method:

```python
features = ['beta', 'se', 'p_value', 'maf', 'your_new_feature']
```

### Adjusting the Autoencoder Architecture

Modify the encoder and decoder architectures in the `GWASAutoencoder` class to match your data complexity:

```python
self.encoder = nn.Sequential(
    nn.Linear(input_dim, your_dim_1),
    nn.ReLU(),
    nn.Linear(your_dim_1, your_dim_2),
    nn.ReLU(),
    nn.Linear(your_dim_2, encoding_dim)
)
```

## Limitations

- Assumes GWAS summary statistics are properly quality controlled
- Requires sufficient memory to load entire dataset
- Clustering results may vary between runs due to random initialization
- Best suited for datasets with at least 1000 SNPs
