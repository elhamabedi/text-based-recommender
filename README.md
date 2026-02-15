## Text-Based Recommender System Using Document Embeddings

A text-based recommender system that predicts restaurant ratings from user reviews using advanced word and document embeddings. Implemented as part of Natural Language Processing course.

##### Assignment Overview
This project implements a simplified collaborative filtering recommender system that predicts user ratings (1–5 stars) for restaurants based solely on textual reviews. The system leverages multiple embedding techniques to capture semantic meaning from text and employs regression models to predict ratings.

##### Key Features
###### Dataset Processing: Extracts a high-quality subset from Yelp Open Dataset with:
Users who reviewed ≥100 restaurants
Restaurants reviewed by ≥1,000 users
Chronological train/validation/test split (20k/1,962/1,962 reviews)
###### Embedding Techniques:
Word2Vec (Skip-gram & CBOW)
FastText (with subword information for OOV handling)
Doc2Vec (Distributed Memory & Distributed Bag of Words)
###### Document Representation:
Word embedding aggregation (average/sum)
Direct paragraph vector learning
###### Hyperparameter Optimization:
Optuna-based tuning for 6 embedding configurations
Optimized parameters: vector size, window, epochs, min_count, learning rate, max_iter, max_depth
###### Comprehensive Evaluation:
Regression metrics: R², MAE, RMSE, MAPE, CCC
Ranking metrics: NDCG@k (k=10,20,50), Spearman ρ, Kendall τ
Pearson correlation for linear relationship analysis

##### Project Structure
```
Text-Based Recommender/
├── dataset/                    # Processed datasets and results
│   ├── filtered_users.csv      # Users meeting criteria (≥100 reviews)
│   ├── filtered_restaurants.csv # Restaurants meeting criteria (≥1,000 reviewers)
│   ├── filtered_reviews.csv    # Complete filtered review dataset
│   ├── train_reviews.csv       # Training set (oldest 20k reviews)
│   ├── val_reviews.csv         # Validation set
│   ├── test_reviews.csv        # Test set (most recent reviews)
│   ├── hyperparameter_results.csv # Optimal hyperparameters for all 6 settings
│   └── evaluation_results.csv  # Comprehensive evaluation metrics
├── figures/                    # Generated visualizations
│   ├── param_importance_*.png  # Hyperparameter importance plots
│   ├── r2_bar_plot.png         # R² comparison across settings
│   └── ndcg_line_plot.png      # NDCG@k performance visualization
├── models/                     # Saved embedding models (optional)
│   ├── word2vec_optimized.model
│   └── fasttext_optimized.model
├── Code.ipynb                  # Main implementation notebook
├── README.md                   # This file
```

##### References
1.Mikolov, T., et al. (2013). Distributed representations of words and phrases and their compositionality. NIPS.


2.Bojanowski, P., et al. (2017). Enriching word vectors with subword information. TACL.


3.Le, Q. & Mikolov, T. (2014). Distributed representations of sentences and documents. ICML.


4.Hasanzadeh, S., et al. (2020). Review-based recommender systems: A proposed rating prediction scheme using word embedding representation of reviews. The Computer Journal.



Note: This is an academic implementation for educational purposes. The Yelp dataset is used under Yelp's Academic Dataset License Agreement.
