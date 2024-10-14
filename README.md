# SubGWAS

SubGWAS is a Python library for analyzing Genome-Wide Association Study (GWAS) summary statistics and detecting novel subtypes using various machine learning and deep learning clustering algorithms. This library provides a flexible and efficient way to process GWAS data, perform clustering analysis, and visualize results.

## Features

- Automatic detection of GWAS summary statistic column names
- Preprocessing of GWAS data, including effect size computation and p-value transformation
- Implementation of multiple clustering algorithms (K-means, DBSCAN, Gaussian Mixture Models)
- Integration with PyTorch for GPU-accelerated computations (Deep learning)
- Visualization tools for clustering results and Manhattan plots
- Flexible handling of additional GWAS data columns

## Installation

You can install SubGWAS using pip:

```
pip install subgwas
```

## Quick Start

Here's a basic example of how to use SubGWAS:

```python
from subgwas import GWASDataLoader, GWASPreprocessor, GWASClustering
from subgwas.visualization import plot_clusters, plot_manhattan

# Load and preprocess data
loader = GWASDataLoader('path/to/gwas_summary_stats.txt')
data = loader.load_data()
data = loader.filter_snps(data)

preprocessor = GWASPreprocessor(data)
preprocessor.compute_effect_sizes()
preprocessor.compute_log_p_values()
preprocessor.encode_chromosomes()
preprocessor.scale_features()
feature_matrix = preprocessor.prepare_feature_matrix()
snp_ids = preprocessor.get_snp_ids()

# Perform clustering
clustering = GWASClustering(n_clusters=3)
kmeans_labels = clustering.kmeans_clustering(feature_matrix)

# Visualize results
plot_clusters(feature_matrix, kmeans_labels, 'K-means Clustering')
plot_manhattan(data, 'Manhattan Plot')
```

This README provides a comprehensive overview of the GWASCluster library, including:

1. A brief description of the library and its purpose
2. Key features
3. Installation instructions
4. A quick start guide with a basic usage example
5. Links to further documentation
6. Information on how to contribute
7. License information
8. Citation information for academic use
9. Contact information for support
