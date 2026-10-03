# Text Clusterer

Cluster UK English documents using TF-IDF and KMeans.


## Overview
- Text documents are converted into **TF-IDF vectors**:
  $$
  \text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log \frac{N}{DF(t)}
  $$

- **KMeans clustering** partitions the vectors into `k` clusters by minimising:
  $$
  \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2
  $$
  where \( mu_i \) is the centroid of cluster \( C_i \).

- Cluster assignments and silhouette score (`0`–`1`, higher is better) are computed. 
 

## Run
```bash
pip install -r requirements.txt
```

```bash
python3 text_clusterer.py
```