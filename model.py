# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# %% [markdown]
# Data Loading

# %%
try:
    df = pd.read_csv("coin_Dogecoin.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'coin_Dogecoin.csv' was not not found")
df

# %% [markdown]
# Data Preprocessing

# %%
# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)


# %%
# Convert the "date" column to datetime object
df["date"] = pd.to_datetime(df["date"])

# %%
# Sort the data by date to maintain time series order
df = df.sort_values(by="date")

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %%
# Calculate "days-_since_start" as simple temporal feature
# This feature helps the model capture time-based trends in the price
df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

# %%
print(f"Data shape after cleaning: {df.shape}")

# %%
print("Columns available for modeling:",df.columns.tolist())

# %% [markdown]
# Visualization before training

# %%
plt.figure(figsize=(10,6))
# Create a scatter plot to show how the closing price (target) has changed over time
sns.scatterplot(x="days_since_start", y="close", data=df,color="red")

# Add trend line for better visual understanding of the overall price movement
sns.regplot(x="days_since_start",y="close",data=df,scatter=False,color="blue")

plt.title("Dodgecoin Closing Price Over Time", fontsize=16)
plt.xlabel("Days Since Start of Data Collection", fontsize=12)
plt.ylabel("Closing Price (USD)", fontsize=12)
plt.grid(True,linestyle="--",alpha=0.7)
plt.show()

# %% [markdown]
# Feature Engineering

# %%
# Select the features (X) and target variable (y)
# We'll use the "high","low","open" and "volume" to predict the "close" price

features = ["high","low","open","volume"]
X = df[features]
y = df["close"]

# %% [markdown]
# Data Splitting and Scaling

# %%
# Split the data into training (80%) and testing (20%) sets
# random_state ensures reproducibilty
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize the StandardScaler for feature scaling
# Scaling is crucial for regularized models like Lasso and Ridge Regression
scaler = StandardScaler()

# Fit the scaler only on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# %% [markdown]
# Model Training and Evaluation

# %%
#  Dictionary to hold all models and a list to store results
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0,random_state=42),
    "Lasso Regression": Lasso(alpha=0.01,random_state=42,max_iter=2000),
    "ElasticNet": ElasticNet(alpha=0.01,l1_ratio=0.5,random_state=42,max_iter=2000)
}
results = []

# Iterate through each model
for name, model in models.items():
    # Train the model using the scaled training data
    model.fit(X_train_scaled,y_train)

    # Make predictions on the scaled test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate performance using the standard regression metrics
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    # Store the results for comparison
    results.append({
        "Model":name,
        "R-squared":r2,
        "Mean Squared Error":mse,
        "Mean Absolute Error": mae,
        "Predictions":y_pred
    })

    # Print the summary of the model's performance
    print(f"{name} Results:")
    print(f"R-squared: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    results_df = pd.DataFrame(results)

    # Identify the best model on the R-squared (highest R2 indicated best fit)
    best_model_name = results_df.loc[results_df["R-squared"].idxmax()]["Model"]
    best_model = models[best_model_name]
    print(f"Best Performing Model: {best_model_name}")

# %% [markdown]
# Visualization after training

# %%
# Get the row of the best performing model
best_model_row = results_df.loc[results_df["R-squared"].idxmax()]
best_predictions = best_model_row["Predictions"]
best_model_name = best_model_row["Model"]

plt.figure(figsize=(12,7))

# Plot the actual closing price from the test set
plt.plot(y_test.values,label="Actual Closing Price",color="blue",linewidth=2)

# Plot the predicted closing prices from the best model
plt.plot(best_predictions,label=f"Predicted Closing Price ({best_model_name})",color="red",linestyle="--",linewidth=2)

plt.title(f"Actual vs Predicted Dodgecoin Closing Prices on the Test Set ({best_model_name})",fontsize=16)
plt.xlabel("Test Sample Index",fontsize=12)
plt.ylabel("Closing Price (USD)",fontsize=12)
plt.legend()
plt.grid(True,linestyle=":",alpha=0.6)
plt.show()


