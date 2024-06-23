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

## Usage

### Prerequisites

- Python 3.x
- Required Python libraries: `requests`, `zipfile`, `pandas`, `json`, `sklearn`, `joblib`, `graphviz`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cricket-match-predictor.git
   cd cricket-match-predictor

