# Self-Supervised GNNs for Recommendation under Data Sparsity

This repository contains the implementation and experimental analysis for a Deep Learning course project on **Self-Supervised Graph Neural Networks for Recommendation under Data Sparsity**.

The project investigates whether self-supervised learning techniques can improve graph-based recommender systems, especially when user-item interaction data are sparse.

## Methods

The following models and methods are implemented:

- **BPR-MF**  
  Matrix Factorization baseline trained with Bayesian Personalized Ranking loss.

- **LightGCN**  
  Graph-based recommendation model using message passing on the user-item graph.

- **Edge Dropout SSL**  
  Self-supervised graph augmentation method using randomly dropped edges.

- **SimGCL-style Embedding Perturbation**  
  Contrastive learning method that adds noise directly to node embeddings.

- **Popularity-Aware Edge Dropout**  
  Proposed adaptive edge dropout method based on item popularity.

- **Degree-Adaptive Noise Injection**  
  Proposed adaptive embedding perturbation method where the noise magnitude depends on node degree.

- **Combined Adaptive SSL**  
  Combination of popularity-aware edge dropout and degree-adaptive noise injection.

## Datasets

Experiments were conducted on:

- MovieLens 100K
- MovieLens 1M

Ratings greater than or equal to 4 were treated as positive interactions.

## Evaluation Protocol

The project uses a leave-one-out evaluation setting:

- One positive interaction per user is held out for testing.
- All remaining positive interactions are used for training.
- For each test user, the true positive item is ranked against 99 randomly sampled negative items.

Evaluation metrics:

- HR@10
- Recall@10
- NDCG@10

## Main Findings

- Self-supervised learning improves the performance of LightGCN on MovieLens 100K.
- Edge Dropout SSL achieves the best performance on the full MovieLens 100K dataset.
- Embedding perturbation methods such as SimGCL and Adaptive Noise are more robust under moderate sparsity.
- Degree-Adaptive Noise achieves the best overall performance on MovieLens 1M.
- SimGCL and Adaptive Noise provide a better performance-runtime trade-off compared to graph augmentation methods.
