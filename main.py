import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Added for Autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Step 1 : Load the dataset
# Make sure 'creditcard.csv' is in the correct path
try:
    data = pd.read_csv(r"creditcard.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please check the file path.")
    # Exiting if the data isn't found, as the rest of the script depends on it.
    exit()

# Step 2: Explore the class distribution
print("Original Class distribution:", Counter(data['Class']))

# Step 3: Split data into features (X) and target (y)
x = data.drop('Class', axis=1)
y = data['Class']

# Step 4: Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Apply SMOTE to scaled training data
smote = SMOTE(random_state=42)
x_train_bal, y_train_bal = smote.fit_resample(x_train_scaled, y_train)
print("Training class distribution after SMOTE and scaling:", Counter(y_train_bal))


# --- ADDITION 1: Autoencoder Implementation ---
print("\n--- Starting Autoencoder Model Training ---")

# The autoencoder is trained only on non-fraudulent (normal) data from the training set
x_train_normal = x_train_scaled[y_train == 0]

# Build the Autoencoder model architecture
input_dim = x_train_normal.shape[1]
encoding_dim = 14  # This is a hyperparameter you can tune

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh")(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile and train the Autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# The model learns to reconstruct normal data
autoencoder.fit(x_train_normal, x_train_normal,
              epochs=10, # Increased epochs for better learning
              batch_size=32,
              shuffle=True,
              validation_split=0.2,
              verbose=1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])

print("--- Autoencoder Training Finished ---")

# Evaluate the Autoencoder on the entire test set
predictions_ae = autoencoder.predict(x_test_scaled)
mse = np.mean(np.power(x_test_scaled - predictions_ae, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})

# Set a threshold for flagging fraud
# Transactions with a high reconstruction error are considered anomalies (fraud)
threshold = np.percentile(error_df.reconstruction_error, 99) # Flag the top 1% of errors
y_pred_ae = (error_df.reconstruction_error > threshold).astype(int)

print("\nAutoencoder Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ae))
print("Classification Report:\n", classification_report(y_test, y_pred_ae, zero_division=0))


# --- Original Script's XGBoost Model ---
# This part is from your original script, for context
print("\n--- Training XGBoost Model (from original script) ---")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# Using the balanced data for training
xgb_model.fit(x_train_bal, y_train_bal)
y_pred_xgb = xgb_model.predict(x_test_scaled)
print("\nXGBoost Results (on balanced data):")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))


# --- ADDITION 2: Simulate Real-Time Detection ---
print("\n--- Simulating Real-Time Detection (with XGBoost) ---")
# We will use the trained XGBoost model to simulate predictions on a small stream of data.
# Let's take the first 5 transactions from the test set.
subset_x = x_test_scaled[:5]
subset_y = y_test[:5]

for i in range(len(subset_x)):
    # In a real-time scenario, you'd process one transaction at a time
    transaction = subset_x[i].reshape(1, -1) # Model expects a 2D array
    
    # Make a prediction with the trained model
    prediction = xgb_model.predict(transaction)
    proba = xgb_model.predict_proba(transaction)[0][1] # Probability of being fraud
    
    print(f"\nTransaction #{i+1}:")
    print(f"  - Model Prediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
    print(f"  - Fraud Confidence: {proba:.4f}")
    print(f"  - Actual Class: {'Fraud' if list(subset_y)[i] == 1 else 'Legitimate'}")
