# Best Runs with Hasse Metrics (CSV Column Definitions)

This directory contains `best_runs_with_hasse_metrics.csv`, which aggregates experiment results with structural and feature-based topological metrics. These metrics are calculated on **Hasse graphs**—subgraphs representing specific topological neighborhoods within the data complexes.

## Column Naming Convention

Most metric columns follow the pattern:  
`{neighborhood}_{metric}_{stat}`

### 1. Neighborhoods
Neighborhoods define how cells (nodes, edges, faces, etc.) in the topological complex are connected:
- **`up_adjacency-k`**: Connectivity between cells of rank $k$ through a shared cell of rank $k+1$.
- **`down_adjacency-k`**: Connectivity between cells of rank $k$ through a shared cell of rank $k-1$.
- **`up_incidence-k`**: Connectivity between cells of rank $k$ and cells of rank $k+1$ (inter-rank).
- **`down_incidence-k`**: Connectivity between cells of rank $k$ and cells of rank $k-1$ (inter-rank).

### 2. Metrics

#### Structural Metrics
Calculated based on the connectivity of the Hasse graph:
- **`spectral_radius`**: The largest eigenvalue of the adjacency matrix. Indicates graph expansion.
- **`spectral_gap`**: Difference between the largest and second-largest eigenvalues of the adjacency matrix. Related to expansion and connectivity.
- **`degree_assortativity`**: Pearson correlation coefficient of degrees between connected nodes. Positive values indicate nodes connect to others with similar degrees.
- **`effective_diameter`**: The 90th percentile of all-pairs shortest path lengths in the Largest Connected Component (LCC).
- **`algebraic_connectivity`**: The smallest non-zero eigenvalue of the normalized Laplacian (Fiedler value). Measures how well-connected the graph is.
- **`kirchhoff_index`**: The sum of the reciprocals of the non-zero eigenvalues of the Laplacian matrix. Also known as the total resistance distance.
- **`clustering_coefficient`**: Measure of the degree to which nodes in a graph tend to cluster together.

#### Feature-based Metrics
Calculated using node features ($x$) and/or labels ($y$):
- **`dirichlet_energy`**: $x^T L x$, where $L$ is the Laplacian. Measures the "smoothness" or variation of features across the graph topology.
- **`adjusted_homophily`**: Edge homophily ($h$) adjusted for the expected homophily of a random labeling. Measures how much more likely connected nodes are to share the same label than by chance.

### 3. Statistics (`stat`)
Since metrics are calculated per graph in a dataset, they are aggregated as:
- **`min`**: Minimum value across all graphs in the dataset.
- **`max`**: Maximum value across all graphs in the dataset.
- **`mean`**: Average value across all graphs in the dataset.

---

## Experiment Metadata
- **`dataset_name`**: The name of the dataset used in the experiment.
- **`config.model.model_domain`**: The topological domain of the model (e.g., graph, simplicial, cell, hypergraph).
- **`score` / `score_std`**: The primary performance metric (e.g., accuracy, ROC-AUC) and its standard deviation across runs.
- **`val_score`**: The performance on the validation set.
- **`config.*`**: Various hyperparameters used during the training run (e.g., `lr`, `batch_size`, `num_layers`).
