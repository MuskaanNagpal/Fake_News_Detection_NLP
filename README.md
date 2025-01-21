Fake News Detection Using NLP and Machine Learning
This project focuses on building a robust and scalable system to classify news articles as either fake or real using Natural Language Processing (NLP) techniques and machine learning models. The solution aims to combat misinformation by automating the detection of fake news with high accuracy.

Objectives
Classify News Articles: Identify whether a given news article is fake or real.
Develop a Practical Pipeline: Create an end-to-end NLP pipeline for text preprocessing, feature extraction, model training, evaluation, and prediction.
Optimize Model Performance: Compare multiple machine learning models, tune hyperparameters, and identify the best-performing model for fake news classification.
Deploy Interactive Solution: Enable users to input a news article and predict its classification using a custom prediction function.

Key Features
Data Preprocessing:
Cleaned and preprocessed news titles using NLP techniques:
Removal of special characters, stopwords, and case normalization.
Tokenization and stemming using PorterStemmer.
Built a Bag-of-Words (BoW) model with CountVectorizer and n-grams for feature extraction (unigrams, bigrams, trigrams).

Exploratory Data Analysis (EDA):
Visualized the distribution of fake vs. real news in the dataset.
Displayed insights into the data composition (e.g., null values, feature types).

Machine Learning Models:
Implemented and compared two machine learning algorithms:
Multinomial Naive Bayes: Achieved an accuracy of 90.59% after hyperparameter tuning (alpha = 0.3).
Logistic Regression: Achieved an accuracy of 93.63% with optimized hyperparameters (C = 0.8).
Evaluated models using metrics such as accuracy, precision, recall, and confusion matrices.

Hyperparameter Tuning:
Fine-tuned the alpha parameter in Naive Bayes and C parameter in Logistic Regression to improve accuracy and generalization.

Custom Prediction Functionality:
Developed a function to predict whether a new article is real or fake.
Validated the function with sample inputs to demonstrate accuracy and interactivity.

Confusion Matrix and Performance Analysis:
Generated detailed confusion matrices and heatmaps to analyze classification performance.
Showcased precision and recall trade-offs for fake and real news.
