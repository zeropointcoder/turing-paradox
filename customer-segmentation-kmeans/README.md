# Customer Segmentation K-means

Segments customers into meaningful groups using K-means clustering on behavioural features.


## Overview
- Synthetic customer data is generated locally using numerical distributions.

- Each customer is represented as a vector:
  
    `x = (age, income, spending)`

- `K-means` clustering partitions the data into *k* clusters by minimising:

    `J = Σ Σ || xᵢ − μⱼ ||²`

    where:
    - *μⱼ* is the centroid of cluster *j*
    - *xᵢ* is a customer assigned to that cluster

- Customers are assigned to the nearest centroid using `Euclidean` distance.

- Cluster quality is measured using the `Silhouette Score`:

    `s = (b − a) / max(a, b)`

- Results are visualised using `income` vs `spending` behaviour.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```