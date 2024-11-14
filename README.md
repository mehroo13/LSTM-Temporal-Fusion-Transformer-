Temporal Fusion Transformer (TFT) Model for Streamflow Prediction and Climate Scenario Forecasting
Overview

This repository contains a Python-based implementation of a Temporal Fusion Transformer (TFT) model to predict streamflow using historical climate data and generate future streamflow predictions under two Shared Socioeconomic Pathway (SSP) scenarios: SSP2-4.5 and SSP5-8.5. The TFT model is designed to leverage temporal patterns and feature dependencies effectively by combining dynamic climate variables and lagged streamflow data.
Project Details

    Objective: Predict streamflow based on past climate and hydrological data and assess future streamflow patterns under different climate scenarios.
    Data Sources: Historical climate data (Monthly data.xlsx) and climate scenario projections for SSP2-4.5 (SSP245.xlsx) and SSP5-8.5 (SSP585.xlsx).
    Approach:
        Variable Selection Network (VSN): Identifies relevant features using a Gated Residual Network (GRN) for each feature.
        Temporal Fusion Transformer (TFT): Integrates VSN, LSTM layers, and GRN to capture temporal dependencies and complex interactions.
        Predictions: Evaluate the model on a test set and predict future streamflow based on SSP scenario data.

Key Files

    Data Preparation:
        Preprocesses climate data by adding lagged streamflow features for improved temporal insights.
        Scales the features and streamflow target variable using MinMaxScaler.

    Model Definition:
        Variable Selection Network (VSN): Selects features dynamically based on Gated Residual Networks (GRNs) and a sigmoid-based feature selection layer.
        Temporal Fusion Transformer (TFT): Core model for streamflow prediction, combining VSN, LSTM, and GRN layers.

    Training and Evaluation:
        Splits data into training and testing sets (70/30 split).
        Trains the model using mean absolute error (MAE) as the loss function.
        Evaluates model performance using metrics like RMSE, MAE, and R².

    Future Streamflow Predictions:
        Generates future predictions under SSP2-4.5 and SSP5-8.5 scenarios.
        Stores results in separate Excel files for easy access and plotting.

Code Usage

    Environment Setup: Ensure all required libraries are installed (NumPy, Pandas, TensorFlow, Scikit-learn, Matplotlib).
    Data Loading: Place Monthly data.xlsx, SSP245.xlsx, and SSP585.xlsx in the designated file paths.
    Run Model: Execute the script to preprocess data, train the TFT model, and generate predictions.
    Visualize Results: Output includes two main visualizations:
        Actual vs. Predicted streamflow for training and test sets.
        Future streamflow projections under SSP scenarios.

Results and Output Files

    Streamflow_Prediction_Results_TFT.xlsx: Actual vs. predicted streamflow for training and testing sets.
    SSP245_Future_Predictions.xlsx & SSP585_Future_Predictions.xlsx: Future streamflow predictions for SSP2-4.5 and SSP5-8.5.
    streamflow_prediction_plot.png: Plot showing actual vs. predicted streamflow.
    future_streamflow_predictions.png: Visualization of future streamflow projections under SSP scenarios.

Requirements

    Python 3.8+
    TensorFlow 2.x
    NumPy, Pandas, Scikit-learn, Matplotlib

Future Improvements

    Model Tuning: Experiment with different architectures or hyperparameters for improved accuracy.
    Additional Scenarios: Integrate other SSP scenarios or datasets.
    Interpretability: Implement explainability methods for deeper insights into feature importance and temporal patterns.

Citation

If you use this code in your research, please cite the appropriate sources and models that contributed to this work.
