# GWAS Subtype Discovery

A tool for discovering genetic subtypes using unsupervised clustering on GWAS summary statistics. This project includes data simulation and analysis components to identify potential genetic subtypes in complex traits.

## Overview

This repository contains two main components:
1. GWAS data simulation script that generates synthetic GWAS summary statistics with hidden subtype structure
2. Subtype discovery analysis using UMAP dimensionality reduction and HDBSCAN clustering

## Requirements

```bash
pip install hdbscan umap-learn scikit-learn pandas numpy matplotlib scipy nbformat
```

## Usage

### 1. Data Simulation

Run `GWAS_Data_Simulation.ipynb` first to generate simulated GWAS data. This notebook:
- Generates synthetic GWAS summary statistics
- Creates subtype-specific genetic effects
- Simulates realistic LD patterns
- Saves data to 'simulated_gwas.csv'

### 2. Subtype Analysis

Run `GWAS_Subtype_Analysis.ipynb` to perform the clustering analysis. This notebook:
- Loads the simulated (or real) GWAS data
- Preprocesses the data
- Performs dimensionality reduction using UMAP
- Identifies subtypes using HDBSCAN clustering
- Visualizes results and generates detailed statistics

## Features

- Automatic detection of number of subtypes
- No need to pre-specify number of clusters
- Handles large-scale GWAS data
- Robust to noise and outliers
- Comprehensive visualization of results
- Detailed statistical analysis of identified subtypes

## Parameters

Key parameters that can be adjusted:

### Data Simulation
- `n_variants`: Number of genetic variants to simulate (default: 100,000)
- `n_causal`: Number of causal variants (default: 1,000)
- `n_subtypes`: Number of true underlying subtypes (default: 3)

### Subtype Discovery
- `min_cluster_size`: Minimum number of variants per cluster (default: 1,000)
- `min_samples`: Minimum samples for core points (default: 20)
- `cluster_selection_epsilon`: Distance threshold for cluster selection (default: 0.7)

## Output

The analysis generates:
1. Cluster assignments for each variant
2. Statistical summaries of each identified subtype
3. Visualizations including:
   - UMAP embedding plot
   - Manhattan-style plot colored by cluster
   - Effect size distributions

## Example Results

For each identified subtype, the tool provides:
- Cluster size
- Mean and median effect sizes
- Minimum p-value
- Number of significant variants
- Top variants
- Chromosomal distribution
- Detailed visualizations

## Notes

- The simulation parameters can be adjusted to create different genetic architectures
- The clustering parameters can be tuned based on your specific needs
- The tool can be used with real GWAS data by skipping the simulation step
