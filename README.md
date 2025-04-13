# netflix_project
Big Data Project

Introduction:
This project implements a hybrid recommender system combining multiple collaborative filtering approaches to address different recommendation scenarios. The system uses both user-based and item-based KNN methods for rating prediction and top-N recommendations, Singular Value Decomposition (SVD) for latent factor modeling, and implicit feedback techniques for simulating and handling non-explicit user interactions.

Data:
The DataFrame was extracted from Cornac, a collaborative filtering library. It contains entries with three key columns: user, item, and rating. Each row represents a user's interaction with an item, where the rating reflects the level of preference or feedback provided by the user.

Key Objectives:
- Implement and compare user-based (U-KNN) and item-based (I-KNN) collaborative filtering for rating prediction
- Develop SVD-based matrix factorization to uncover latent user-item relationships
- Create implicit feedback recommendations for scenarios with binary data
- Generate top-N recommendations for users
- Evaluate model performance using appropriate metrics (MAE, RMSE, Precision@K, Recall@K)
- Provide insights of differences among different models