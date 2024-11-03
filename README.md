# GrapeQuality
# Grape Variety Prediction Using Neural Networks

This project uses a feed-forward neural network to predict the variety of grapes based on their characteristics, such as sugar content, acidity, and sun exposure. The dataset includes various grape varieties with multiple features that influence grape quality. The model is trained on a labeled dataset to classify grapes accurately.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)

## Project Overview

This project builds a neural network to classify grape varieties based on features related to grape quality and environmental factors. The goal is to achieve accurate predictions by training a model on the provided dataset.

Key objectives:
1. Preprocess the dataset by encoding categorical variables and scaling numerical features.
2. Implement a feed-forward neural network to classify grape varieties.
3. Evaluate the model's performance and tune hyperparameters for improved accuracy.

## Dataset

The dataset used in this project includes the following features:
- `quality_score`: Numerical quality rating.
- `sugar_content_brix`: Sugar content in degrees Brix.
- `acidity_ph`: Acidity level (pH).
- `cluster_weight_g`: Weight of the grape cluster in grams.
- `berry_size_mm`: Average berry size in millimeters.
- `sun_exposure_hours`: Hours of sun exposure.
- `soil_moisture_percent`: Soil moisture as a percentage.
- `rainfall_mm`: Rainfall in millimeters.
- `region`: Categorical, e.g., "Barossa Valley", "Napa Valley".
- `quality_category`: Categorical, e.g., "High", "Medium".

The target variable is `variety`, representing different types of grapes.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/grapequality.git
   cd grapequality

2. Install the required dependencies:

    ```bash
    pandas
    numpy
    scikit-learn
    tensorflow

## Usage

1. Data Processing: The dataset is preprocessed to handle categorical variables, scale numerical features, and split into training and testing sets.
2. Model training: Train the neural network by running the main script. You can specify the number of epochs and batch size.
3. Predition: Use the `predict_variety` function to classify a new grape sample based on its features. Example usage:

    ```bash
    from model import predict_variety

    sample_features = {
        'quality_score': 3.0,
        'sugar_content_brix': 22.0,
        'acidity_ph': 3.2,
        'cluster_weight_g': 250.0,
        'berry_size_mm': 18.0,
        'sun_exposure_hours': 10.0,
        'soil_moisture_percent': 50.0,
        'rainfall_mm': 300.0,
        'region_Barossa Valley': 1,
        'region_Loire Valley': 0,
        'region_Napa Valley': 0,
        'quality_category_High': 0,
        'quality_category_Low': 0,
        'quality_category_Medium': 1,
        'quality_category_Premium': 0
}

    predicted_variety = predict_variety(sample_features)
    print("Predicted variety:", predicted_variety)

## Model Architecture

The neural network is a feed-forward architecture with the following specifications:

Input Layer: Takes preprocessed features.

Hidden Layers: Two hidden layers with ReLU activation.

Output Layer: Softmax activation for multi-class classification.

Hyperparameters such as batch size, learning rate, and number of epochs can be modified in problem_set3_490.py.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any enhancements, fixes, or additional features. I made this for a university project, so use it as you like <3


