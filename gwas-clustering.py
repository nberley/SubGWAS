import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd

class GWASAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(GWASAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class GWASClusterAnalysis:
    def __init__(self, encoding_dim=32, n_clusters_range=(2, 10)):
        self.encoding_dim = encoding_dim
        self.n_clusters_range = n_clusters_range
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def preprocess_data(self, gwas_data):
        """Preprocess GWAS summary statistics."""
        # Extract relevant features
        features = ['beta', 'se', 'p_value', 'maf']  # Add/modify features as needed
        X = gwas_data[features].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0)
        
        # Log transform p-values
        X[:, features.index('p_value')] = -np.log10(X[:, features.index('p_value')])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return torch.FloatTensor(X_scaled)
    
    def train_autoencoder(self, X, epochs=100, batch_size=32):
        """Train autoencoder for dimensionality reduction."""
        input_dim = X.shape[1]
        model = GWASAutoencoder(input_dim, self.encoding_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        X = X.to(self.device)
        
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                
                optimizer.zero_grad()
                encoded, decoded = model(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        return model
    
    def perform_clustering(self, encoded_data):
        """Perform clustering and determine optimal number of clusters."""
        best_score = -1
        best_n_clusters = 2
        best_labels = None
        
        encoded_np = encoded_data.detach().cpu().numpy()
        
        # Try different numbers of clusters
        for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            # Initialize cluster centers using k-means++
            centers = encoded_np[np.random.choice(encoded_np.shape[0], n_clusters, replace=False)]
            
            for _ in range(50):  # Maximum iterations for k-means
                # Assign points to nearest center
                distances = np.sqrt(((encoded_np[:, None] - centers) ** 2).sum(axis=2))
                labels = distances.argmin(axis=1)
                
                # Update centers
                new_centers = np.array([encoded_np[labels == k].mean(axis=0) for k in range(n_clusters)])
                
                if np.allclose(centers, new_centers):
                    break
                    
                centers = new_centers
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(encoded_np, labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
        
        return best_labels, best_n_clusters, best_score
    
    def analyze_clusters(self, gwas_data, cluster_labels):
        """Analyze characteristics of identified clusters."""
        gwas_data['cluster'] = cluster_labels
        cluster_stats = {}
        
        for cluster in range(len(np.unique(cluster_labels))):
            cluster_data = gwas_data[gwas_data['cluster'] == cluster]
            stats = {
                'size': len(cluster_data),
                'mean_beta': cluster_data['beta'].mean(),
                'mean_p_value': cluster_data['p_value'].mean(),
                'mean_maf': cluster_data['maf'].mean(),
                'top_snps': cluster_data.nsmallest(10, 'p_value')['snp'].tolist()
            }
            cluster_stats[f'cluster_{cluster}'] = stats
        
        return cluster_stats

def main(gwas_file):
    """Main function to run the analysis."""
    # Load GWAS data
    gwas_data = pd.read_csv(gwas_file)
    
    # Initialize analyzer
    analyzer = GWASClusterAnalysis()
    
    # Preprocess data
    X = analyzer.preprocess_data(gwas_data)
    
    # Train autoencoder
    model = analyzer.train_autoencoder(X)
    
    # Get encoded representations
    encoded_data, _ = model(X)
    
    # Perform clustering
    cluster_labels, n_clusters, silhouette = analyzer.perform_clustering(encoded_data)
    
    # Analyze clusters
    cluster_stats = analyzer.analyze_clusters(gwas_data, cluster_labels)
    
    print(f"\nOptimal number of clusters: {n_clusters}")
    print(f"Silhouette score: {silhouette:.3f}")
    
    # Print cluster statistics
    for cluster, stats in cluster_stats.items():
        print(f"\n{cluster.upper()}:")
        print(f"Size: {stats['size']}")
        print(f"Mean beta: {stats['mean_beta']:.3f}")
        print(f"Mean MAF: {stats['mean_maf']:.3f}")
        print("Top SNPs:", ", ".join(stats['top_snps'][:5]))

if __name__ == "__main__":
    main("path_to_your_gwas_summary_stats.csv")
