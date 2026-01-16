# Optimizing the Binary Classification Problem: Fake News Classification
This project studies fake news detection as a binary classification problem, with a primary focus on optimization behavior rather than model architecture. Logistic Regression with L2 regularisation is used as a convex learning objective, enabling a controlled comparison of multiple numerical optimization algorithms applied to the same loss function. Beyond classification accuracy, the project analyzes convergence speed, numerical stability, runtime, and memory efficiency of different optimization methods.

*This is a course project for CMSE 831 Computational Optimization at Michigan State University.

## Table of Contents
1. [Dataset, Preprocessing and Feature Representation](#dataset-preprocessing-and-feature-representation)
2. [Problem Setup](#problem-setup)
3. [Implementation and Results](#implementation-and-results)

## Dataset, Preprocessing and Feature Representation
The [WELFake (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) dataset combines Kaggle, McIntire, Reuters, and BuzzFeed political news. Articles are cleaned to remove HTML artifacts, excessive whitespace, and duplicates. Very short articles are removed to ensure meaningful textual content. Data is split into 80–10–10 train/validation/test sets with stratification.

TF-IDF Vectorization is implemented with 5,000 features, unigrams and bigrams, stop-word removal, and sublinear term frequency scaling. This produces a high-dimensional sparse feature space, well suited for studying optimizer behavior.

## Problem Setup
I minimize the regularized logistic regression loss:
- Objective: Negative log-likelihood + L2 regularization
- Convex and strongly convex (guaranteed unique global optimum)
- Optimization variables: weight vector (5,000D) and bias term

All methods use approximate line search and identical stopping criteria:
- Gradient Descent (GD)
- Non-linear Conjugate Gradient (Polak–Ribiere)
- Non-linear Conjugate Gradient (Fletcher–Reeves)
- Newton’s Method
- Quasi-Newton (L-BFGS)

## Implementation and Results
All optimizers converge to essentially the same solution, achieving ~90% test accuracy and ~0.969 AUC.

Key observations:
- Gradient Descent converges reliably but slowly.
- CG (FR) improves substantially over GD and CG-PR.
- Newton’s Method converges in only a few iterations but is computationally expensive.
- L-BFGS offers the best overall trade-off, reaching near-Newton precision with far lower runtime and memory usage.

The notebook includes Loss vs. iteration plots, Gradient norm vs. iteration (log scale), and 2D projections of optimization paths around the optimum. These plots visually confirm theoretical convergence properties and efficiency differences between first-order, second-order, and quasi-Newton methods.

The project demonstrates that while many optimization algorithms can successfully solve convex machine-learning objectives, their practical efficiency differs dramatically. For high-dimensional logistic regression on sparse text data, L-BFGS emerges as the most effective optimizer, balancing speed, accuracy, and memory usage.
