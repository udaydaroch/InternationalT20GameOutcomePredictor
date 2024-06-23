# Cricket Match Outcome Predictor

## Overview

This project aims to predict the outcome of T20 cricket matches using machine learning, specifically the Random Forest algorithm. By analyzing historical match data, the model predicts the winning team based on various match-related features.

## Inspiration

Having played cricket and being passionate about the sport, I was curious to explore how software and sports can be integrated. This project is an attempt to leverage machine learning to predict the outcomes of cricket matches, demonstrating the intersection of sports analytics and technology.

## Data Source

The data used for this project is sourced from [Cricsheet](https://cricsheet.org/), a free source of structured data of cricket matches. The specific dataset used is the T20 matches in JSON format, which includes detailed information about each match.

## Project Structure

1. **Data Download and Extraction**: The dataset is downloaded from Cricsheet and extracted from a zip file.
2. **Data Loading and Cleaning**: JSON files are loaded, and match-level data is extracted. Matches with no recorded outcome are excluded.
3. **Data Preprocessing**: Categorical variables are encoded using one-hot encoding, and the target variable (match winner) is label-encoded.
4. **Model Training**: A Random Forest classifier is trained on the processed data.
5. **Model Evaluation**: The model's performance is evaluated using accuracy, precision, recall, and F1 score.
6. **Prediction**: The trained model is used to predict the outcome of new matches.
7. **Visualization**: One of the decision trees in the Random Forest is visualized.

## Random Forest Algorithm

The Random Forest algorithm is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and robustness. It works as follows:

1. **Bootstrap Aggregating (Bagging)**: Multiple subsets of the training data are created by sampling with replacement. Each decision tree in the forest is trained on a different subset.
2. **Random Feature Selection**: At each split in the decision tree, a random subset of features is considered. This introduces randomness and helps to de-correlate the trees.
3. **Tree Construction**: Each tree is grown to the maximum possible depth without pruning.
4. **Prediction Aggregation**: For classification tasks, each tree votes for a class, and the majority vote is taken as the final prediction. For regression tasks, the predictions are averaged.

### How It's Used in This Project

In this project, the Random Forest algorithm is used to predict the winner of T20 cricket matches. The model considers features such as the teams playing, the toss winner, the venue, and the gender of the teams. By training on historical match data, the Random Forest model learns patterns and relationships that help it make predictions for new matches.

## Evaluation Metrics

1. **Precision**: High precision indicates that the model is accurate when it predicts a positive class. A low precision means there are many false positives.
2. **Recall**: High recall indicates that the model is good at capturing all the positive instances. A low recall means there are many false negatives.
3. **F1-Score**: The F1-score is a balance between precision and recall. A high F1-score means the model has a good balance between precision and recall.
4. **Support**: This tells us how many actual instances of each class there are in the dataset. It provides context for interpreting the other metrics.

## Limitations

While the model provides useful predictions, there are several limitations to be aware of:

1. **Player Performance**: The model does not account for individual player performance, which can significantly impact the outcome of a match. Factors such as form, injuries, and player matchups are not included in the dataset.
2. **Weather Conditions**: Weather conditions can greatly influence the outcome of a cricket match, but they are not considered in the current model.
3. **Team Composition**: The model does not take into account the specific composition of the teams, such as the presence of key players or the balance between batsmen and bowlers.
4. **Venue Factors**: While the venue is included as a feature, detailed conditions of the pitch and ground are not considered.
5. **Historical Data**: The model is trained on historical data, which means it might not fully capture the dynamics of current team strengths and strategies.
6. **External Factors**: Other factors such as team morale, recent performance trends, and off-field issues are not included in the model.

Addressing these limitations could improve the accuracy and reliability of the predictions. Future work could involve integrating more detailed data on player performance and conditions.

## Files in This Repository

1. `cricket_match_predictor.py`: The main script that downloads data, preprocesses it, trains the model, and evaluates its performance.
2. `random_forest_model.joblib`: The trained Random Forest model saved as a joblib file.
3. `combined_matches.jsonl`: The combined match data saved in JSON Lines format.
4. `tree.dot`: A DOT file for visualizing one of the trees in the Random Forest.

## How to Run

1. Ensure you have Python and the required libraries installed.
2. Download the data by running the script `cricket_match_predictor.py`.
3. The model will be trained, evaluated, and saved as `random_forest_model.joblib`.
4. Use the model to make predictions on new match data as demonstrated in the script.

## Requirements

1. Python 3.x
2. pandas
3. requests
4. scikit-learn
5. joblib
6. graphviz

Install the required libraries using:

```bash
pip install pandas requests scikit-learn joblib graphviz

