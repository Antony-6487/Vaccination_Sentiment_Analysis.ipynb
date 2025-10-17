# Vaccination_Sentiment_Analysis.ipynb
This repository contains my solution for the Zindi
Zindi “To Vaccinate or Not to Vaccinate” — Sentiment Regression Challenge
 It's not a Question”, where the goal is to develop a machine learning model that predicts public sentiment scores on vaccination-related tweets, ranging from -1 (negative) to 1 (positive).
The main evaluation metric is Root Mean Squared Error (RMSE).
Project Overview

Vaccines have played a crucial role in preventing diseases globally. However, misinformation and anti-vaccine sentiment can undermine these efforts. This project focuses on:

Analyzing and cleaning tweet text data

Converting text into numerical features using TF-IDF

Training a Ridge Regression model to predict sentiment scores

Tuning hyperparameters via cross-validation to optimize RMSE

📂 Repository Structure
├── Train.csv              # Training dataset with text and sentiment scores
├── Test.csv               # Test dataset (no labels)
├── SampleSubmission.csv   # Submission format
├── notebook.ipynb         # Colab/Notebook with full pipeline
├── submission.csv         # Final prediction file ready for upload to Zindi
└── README.md              # Project documentation

🧰 Tech Stack

Python 3

pandas, numpy

scikit-learn (TF-IDF, Ridge Regression, Cross-validation)

nltk (Stopwords)

Google Colab / Jupyter Notebook

📊 Model Pipeline

Data Cleaning

Lowercasing, punctuation removal, stopword removal

Feature Engineering

TF-IDF vectorization (1–2 n-grams, up to 10k features)

Modeling

Ridge Regression tuned using 5-fold cross-validation

Final predictions generated on test data
Evaluation

RMSE used for scoring (target ≈ 0.57 RMSE on leaderboard)




Evaluation

RMSE used for scoring (target ≈ 0.43 RMSE on leaderboard)
