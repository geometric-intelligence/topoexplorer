## Statistical Output Columns

Each row in the output files represents the statistical relationship between a specific topological property of a Hasse graph and the normalized performance improvement of the TopoTune model relative to a baseline GNN.

* **Metric**: The name of the topological property extracted from the dataset's Hasse graph.
* **Spearman_Rho**: The Spearman's rank correlation coefficient ($\rho$). This non-parametric measure ranges from -1.0 to 1.0. A value near 1.0 indicates a strong monotonic positive relationship (higher metric value correlates with better relative model performance), while a value near -1.0 indicates a strong monotonic negative relationship.
* **P_Value**: The raw, unadjusted $p$-value associated with the Spearman correlation. It tests the null hypothesis that there is no monotonic relationship between the metric and the model performance.
* **N_samples**: The number of independent observations (datasets) included in the calculation for this specific metric. Metrics with constant values across all datasets or missing data are filtered out.
* **Adj_P_Value**: The $p$-value adjusted for multiple comparisons using the Benjamini-Hochberg False Discovery Rate (FDR) procedure. This accounts for the inflated risk of Type I errors (false positives) when testing hundreds of metrics simultaneously.
* **Significant**: A boolean flag (`True` or `False`) indicating whether the correlation is statistically significant after FDR correction at the $\alpha = 0.05$ threshold.