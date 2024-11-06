# Fake News Detection using NLP and Machine Learning

This project aims to develop a Fake News Detection model using Natural Language Processing (NLP) and Machine Learning techniques to distinguish between real and fake news articles. With a large dataset sourced from Kaggle, the project applies various text processing and machine learning algorithms to enhance the reliability of information veracity in digital media.

# Project Overview

Objective: Accurately classify news content as real or fake using NLP and machine learning.
Dataset: A large Kaggle dataset with 20,800 samples for training and 5,200 samples for testing.


# Key Components

1. Exploratory Data Analysis (EDA):
   - Initial inspection of data structure, distribution of real vs. fake news labels.
   - Visualizations and summary statistics to understand data characteristics.

2. Data Preprocessing:
   -Cleaning: Removing stop words, links, numbers, and irrelevant content.
   - Tokenization: Breaking text into smaller units (tokens).
   - Stemming & Lemmatization: Reducing words to their base forms for better analysis.

3. Word Cloud Visualization:
   - Word clouds for both fake and real news to identify commonly used words in each category.

4. Vectorization:
   - TF-IDF Vectorizer to convert text data into numerical form for model training.

5. Machine Learning Models:
   - Logistic Regression: Achieved 96.63% accuracy.
   - Support Vector Machine (SVM): Achieved 97.98% accuracy.
   - Naive Bayes: Achieved 86.44% accuracy, with a higher tendency to correctly classify fake news.

# Results and Insights

Model Performance:
   - Both Logistic Regression and SVM showed high accuracy and consistency in detecting fake news.
   - Naive Bayes, while slightly less accurate, demonstrated robust identification for fake news.

# Conclusion:
The results affirm the effectiveness of NLP and machine learning in filtering fake news, supporting reliable media consumption.

# File Structure

nlp_project.ipynb: Main notebook containing all code for EDA, pre processing, modeling, and evaluation.



# Usage

Run `nlp_project.ipynb` in a Jupyter Notebook environment to:
1. Load and preprocess the dataset.
2. Train and evaluate the machine learning models.
3. View visualizations and performance metrics for each model.
