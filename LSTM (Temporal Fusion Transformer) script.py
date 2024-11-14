# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load and prepare the dataset
# Assumes the Excel file contains 'Date', 'PCP' (precipitation), 'Temp Max', 'Temp Min', and 'Streamflow' columns
data = pd.read_excel('E:\\New machine learning analysis\\Observed data\\Monthly data.xlsx')
dates = data['Date']  # Save dates for plotting purposes

# Separate input features and target variable
X_dynamic_features = data[['PCP', 'Temp Max', 'Temp Min']].values
y_target = data['Streamflow'].values

# Add lagged streamflow as additional input features (lags of 1 to 12 months)
for lag in range(1, 13):
    data[f'Lag_Streamflow_{lag}'] = data['Streamflow'].shift(lag)

# Drop rows with NaN values created by lagging
data = data.dropna()

# Redefine inputs and target to include lagged streamflow features
X_dynamic_features = data[['PCP', 'Temp Max', 'Temp Min'] + [f'Lag_Streamflow_{i}' for i in range(1, 13)]].values
y_target = data['Streamflow'].values

# Scale features and target to range [0, 1]
scaler_X = MinMaxScaler()
X_dynamic_scaled = scaler_X.fit_transform(X_dynamic_features)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1))

# Reshape input data for LSTM layer to 3D format (samples, timesteps, features)
X_dynamic_scaled = X_dynamic_scaled.reshape((X_dynamic_scaled.shape[0], 1, X_dynamic_scaled.shape[1]))

# Split data into training and testing sets (70% training, 30% testing)
train_percentage = 0.7
X_train, X_test, y_train, y_test = train_test_split(
    X_dynamic_scaled, y_scaled, test_size=(1 - train_percentage), random_state=42)

# Define a custom Gated Residual Network (GRN) layer
class GRN(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GRN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units)
        self.gate = tf.keras.layers.Dense(units, activation='sigmoid')
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        gate_output = self.gate(inputs)
        x = gate_output * x
        x = self.layer_norm(x + inputs)
        return x

# Define a Variable Selection Network with a sigmoid adjustment for feature selection
class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, num_features, num_units):
        super(VariableSelectionNetwork, self).__init__()
        self.grn_layers = [GRN(num_units) for _ in range(num_features)]
        self.selection_layer = tf.keras.layers.Dense(num_features, activation='sigmoid')

    def call(self, inputs):
        feature_outputs = [self.grn_layers[i](inputs[..., i:i + 1]) for i in range(inputs.shape[-1])]
        feature_outputs = tf.stack(feature_outputs, axis=-1)
        selection_weights = self.selection_layer(tf.reduce_mean(feature_outputs, axis=-2))
        selection_weights = tf.expand_dims(selection_weights, axis=-2)
        selected_features = tf.reduce_sum(feature_outputs * selection_weights, axis=-1)
        return selected_features

# Define the Temporal Fusion Transformer (TFT) Model
class TemporalFusionTransformer(tf.keras.Model):
    def __init__(self, num_features, hidden_units):
        super(TemporalFusionTransformer, self).__init__()
        self.variable_selection = VariableSelectionNetwork(num_features, hidden_units)
        self.lstm_layer = tf.keras.layers.LSTM(hidden_units, return_sequences=True)
        self.grn_layer = GRN(hidden_units)
        self.dense_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.variable_selection(inputs)
        x = self.lstm_layer(x)
        x = self.grn_layer(x)
        output = self.dense_output(x)
        return output

# Instantiate and compile the model
num_features = X_dynamic_scaled.shape[2]
hidden_units = 64
tft_model = TemporalFusionTransformer(num_features=num_features, hidden_units=hidden_units)
tft_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')

# Train the model
history = tft_model.fit(X_train, y_train, epochs=1500, batch_size=32, validation_data=(X_test, y_test))

# Make predictions on train and test sets
y_train_pred = tft_model.predict(X_train)
y_test_pred = tft_model.predict(X_test)

# Inverse scale predictions to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))
y_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
train_mae = mean_absolute_error(y_train_actual, y_train_pred)
test_mae = mean_absolute_error(y_test_actual, y_test_pred)
train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

print(f'Training RMSE: {train_rmse}, MAE: {train_mae}, R2: {train_r2}')
print(f'Testing RMSE: {test_rmse}, MAE: {test_mae}, R2: {test_r2}')

# Save predictions and actual values to Excel
results_df = pd.DataFrame({
    'Date': np.concatenate([dates[:len(y_train_actual)], dates[len(y_train_actual):len(y_train_actual) + len(y_test_actual)]]),
    'Actual Streamflow': np.concatenate([y_train_actual.flatten(), y_test_actual.flatten()]),
    'Predicted Streamflow': np.concatenate([y_train_pred.flatten(), y_test_pred.flatten()]),
    'Set': ['Train'] * len(y_train_actual) + ['Test'] * len(y_test_actual)
})
output_file = 'E:\\New machine learning analysis\\Observed data\\Streamflow_Prediction_Results_TFT.xlsx'
results_df.to_excel(output_file, index=False)

# Plot actual vs predicted streamflow for train and test sets
plt.figure(figsize=(14, 7), dpi=300)
plt.subplot(2, 1, 1)
plt.plot(dates[:len(y_train_actual)], y_train_actual, label='Actual Streamflow (Train)', color='blue')
plt.plot(dates[:len(y_train_pred)], y_train_pred, label='Predicted Streamflow (Train)', color='orange')
plt.title('Actual vs Predicted Streamflow (Training Set)')
plt.xlabel('Date')
plt.ylabel('Streamflow')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(dates[len(y_train_actual):len(y_train_actual) + len(y_test_actual)], y_test_actual, label='Actual Streamflow (Test)', color='blue')
plt.plot(dates[len(y_train_pred):len(y_train_pred) + len(y_test_pred)], y_test_pred, label='Predicted Streamflow (Test)', color='orange')
plt.title('Actual vs Predicted Streamflow (Testing Set)')
plt.xlabel('Date')
plt.ylabel('Streamflow')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('E:\\New machine learning analysis\\Observed data\\streamflow_prediction_plot.png', dpi=300)
plt.show()

# Future predictions for SSP245 and SSP585 scenarios
ssp245_data = pd.read_excel('E:\\New machine learning analysis\\Observed data\\SSP 245.xlsx')
ssp585_data = pd.read_excel('E:\\New machine learning analysis\\Observed data\\SSP585.xlsx')

# Add lagged streamflow features to future datasets
for lag in range(1, 13):
    ssp245_data[f'Lag_Streamflow_{lag}'] = ssp245_data['PCP'].shift(lag)
    ssp585_data[f'Lag_Streamflow_{lag}'] = ssp585_data['PCP'].shift(lag)

ssp245_data = ssp245_data.dropna()
ssp585_data = ssp585_data.dropna()

# Extract and scale features
ssp245_X = ssp245_data[['PCP', 'Temp Max', 'Temp Min'] + [f'Lag_Streamflow_{i}' for i in range(1, 13)]].values
ssp585_X = ssp585_data[['PCP', 'Temp Max', 'Temp Min'] + [f'Lag_Streamflow_{i}' for i in range(1, 13)]].values

ssp245_X_scaled = scaler_X.transform(ssp245_X).reshape((ssp245_X.shape[0], 1, ssp245_X.shape[1]))
ssp585_X_scaled = scaler_X.transform(ssp585_X).reshape((ssp585_X.shape[0], 1, ssp585_X.shape[1]))

# Predict future streamflow
ssp245_pred = scaler_y.inverse_transform(tft_model.predict(ssp245_X_scaled).reshape(-1, 1))
ssp585_pred = scaler_y.inverse_transform(tft_model.predict(ssp585_X_scaled).reshape(-1, 1))

# Save predictions to Excel
ssp245_pred_df = pd.DataFrame({'Date': ssp245_data['Date'], 'Predicted Streamflow (SSP245)': ssp245_pred.flatten()})
ssp585_pred_df = pd.DataFrame({'Date': ssp585_data['Date'], 'Predicted Streamflow (SSP585)': ssp585_pred.flatten()})

ssp245_pred_df.to_excel('E:\\New machine learning analysis\\Observed data\\SSP245_Future_Predictions.xlsx', index=False)
ssp585_pred_df.to_excel('E:\\New machine learning analysis\\Observed data\\SSP585_Future_Predictions.xlsx', index=False)

# Plot future predictions for SSP scenarios
plt.figure(figsize=(30, 10), dpi=300)

plt.subplot(2, 1, 1)
plt.plot(ssp245_data['Date'], ssp245_pred.flatten(), label='Predicted Streamflow (SSP245)', color='green')
plt.title('Future Streamflow Predictions (SSP245)')
plt.xlabel('Date')
plt.ylabel('Streamflow')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ssp585_data['Date'], ssp585_pred.flatten(), label='Predicted Streamflow (SSP585)', color='red')
plt.title('Future Streamflow Predictions (SSP585)')
plt.xlabel('Date')
plt.ylabel('Streamflow')
plt.legend()
plt.grid(True)

plt.tight_layout(pad=3.0)
plt.savefig('E:\\New machine learning analysis\\Observed data\\future_streamflow_predictions.png', dpi=300)
plt.show()
