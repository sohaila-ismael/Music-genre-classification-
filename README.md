# Music Genre Classification Project

## Project Overview
This project focuses on classifying music genres using machine learning techniques. The dataset used contains various features related to music tracks, such as 'Distorted Guitar' and 'Metal Frequencies', and the goal is to predict the genre of a track based on these features. The project also explores dimensionality reduction through Principal Component Analysis (PCA) and logistic regression as a classification algorithm.

## Dataset
The dataset used in this project contains multiple features related to music tracks and their respective genres. The dataset is stored in a CSV file named music_dataset_mod.csv.

- Features: Various musical attributes (e.g., 'Distorted Guitar', 'Metal Frequencies').
- Target: The genre of each track.

## Project Structure
1. Data Preprocessing:
   - Handling missing values in the 'Genre' column.
   - Label encoding of categorical values (Genres).
   - Standard scaling of numerical features to ensure better performance with machine learning models.

2. Exploratory Data Analysis (EDA):
   - Visualization of genre distribution using seaborn.
   - Correlation analysis of relevant features.

3. Dimensionality Reduction (PCA):
   - Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset.
   - The explained variance ratio is plotted to determine the optimal number of components.

4. Model Training:
   - Logistic Regression: 
     - Trained on the PCA-transformed data.
     - Evaluated using classification reports and accuracy scores.
   - A comparison between logistic regression models with and without PCA is made to assess the impact of dimensionality reduction.

5. Model Evaluation:
   - Classification report includes metrics like precision, recall, f1-score, and support for each genre.
   - Accuracy score is calculated to evaluate model performance.

6. Handling Unknown Genres:
   - For unknown genres (where the 'Genre' is missing), the trained logistic regression model is used to predict the missing genres after applying PCA.

## Installation
To run this project, the following Python libraries are required:

Bash


pip install pandas numpy scikit-learn matplotlib seaborn

## Usage

1. Load the dataset into a pandas DataFrame.
2. Preprocess the data by handling missing values and scaling the features.
3. Visualize the data using seaborn and matplotlib to understand the genre distribution and feature correlations.
4. Apply PCA to reduce the number of features while retaining important variance.
5. Train a logistic regression model on the transformed dataset.
6. Evaluate the model using classification metrics.
7. Predict the genre of tracks with missing genre information using the trained model.

## Model Performance

- The Logistic Regression model with PCA achieved an accuracy of approximately 49.33%.
- Without PCA, the model achieved an accuracy of approximately 51%.
- While dimensionality reduction helped reduce the complexity of the dataset, it did not significantly improve the model's performance.

## Conclusion 

This project demonstrates the application of machine learning techniques to classify music genres. While PCA helps reduce dimensionality, the logistic regression model showed moderate performance in accurately predicting genres. Further improvements could include experimenting with other machine learning models or feature engineering to increase prediction accuracy.

## Future Improvements 

- Testing other classification algorithms (e.g., Random Forest, SVM).
- Performing more feature engineering to extract better predictors from the music dataset.
- Tuning the PCA components to optimize the trade-off between dimensionality reduction and model accuracy.
