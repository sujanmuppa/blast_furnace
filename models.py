import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load data
df = pd.read_excel('demo3.xlsx')

# Define input features and target variables
features = ['CB_FLOW', 'CB_PRESS', 'CB_TEMP', 'STEAM_FLOW', 'STEAM_TEMP', 'STEAM_PRESS',
            'O2_PRESS', 'O2_FLOW', 'O2_PER', 'PCI', 'ATM_HUMID', 'HB_TEMP', 'HB_PRESS',
            'TOP_PRESS', 'TOP_TEMP1', 'TOP_TEMP2', 'TOP_TEMP3', 'TOP_TEMP4', 'TOP_SPRAY',
            'TOP_TEMP', 'TOP_PRESS_1', 'H2']

target_1hr = 'Next_1hr'
target_2hr = 'Next_2hr'
target_3hr = 'Next_3hr'
target_4hr = 'Next_4hr'

# Split data
X = df[features]
y_1hr = df[target_1hr]
y_2hr = df[target_2hr]
y_3hr = df[target_3hr]
y_4hr = df[target_4hr]

# model for next_1hr
X_train, X_test, y_train, y_test = train_test_split(X, y_1hr, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_1 = xgb.XGBRegressor()
model_1.fit(X_train_scaled, y_train)


print(model_1.n_features_in_)

# model for next_2hr

features += ['Next_1hr']

X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y_2hr, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_2 = xgb.XGBRegressor()
model_2.fit(X_train_scaled, y_train)

print(model_2.n_features_in_)

# model for next_3hr

features += ['Next_2hr']

X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y_3hr, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_3 = xgb.XGBRegressor()
model_3.fit(X_train_scaled, y_train)

print(model_3.n_features_in_)

# model for next_4hr

features += ['Next_3hr']
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y_4hr, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_4 = xgb.XGBRegressor()
model_4.fit(X_train_scaled, y_train)

print(model_4.n_features_in_)

# Save the models
import pickle

model_names = ['model_1', 'model_2', 'model_3', 'model_4']

for name in model_names:
    model = globals()[name]
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
