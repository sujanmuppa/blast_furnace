# load the models 
import pickle
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
# Load the models

models = {}
model_names = ['model_1', 'model_2', 'model_3', 'model_4']

for name in model_names:
    with open(f'{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)

# Predict next_1hr
data = {
    "features": [
        [311727, 3.15, 129, 4, 213, 3.34, 3.2, 7296, 23.08, 32, 24.56, 1060, 2.99, 1.5, 112, 135, 107, 130, 0, 121,2, 3.88],
        [312520, 3.2, 126, 4, 189, 3.39, 3.25, 7916, 23.08, 36, 25.37, 1059, 3.03, 1.5, 123, 139, 117, 132, 0, 128, 2, 4.12],
        [311177, 3.21, 126, 4, 191, 3.4, 3.26, 7922, 23.08, 33, 26.15, 1057, 3.05, 1.5, 117, 138, 126, 129, 0, 128, 1, 4.11],
        [301825, 3.17, 127, 4, 190, 3.36, 3.21, 7914, 23.08, 30, 25.42, 1058, 3.01, 1.49, 132, 155, 150, 139, 0, 144, 1, 4.12],
        [313514, 3.27, 124, 6, 192, 3.48, 3.32, 8411, 23.08, 30, 22.67, 1058, 3.11, 1.5, 130, 151, 149, 133, 0, 141, 2, 4.14],
        [318428, 3.21, 127, 6, 189, 3.43, 3.27, 8096, 23.08, 27, 21.78, 1049, 3.05, 1.49, 137, 145, 124, 140, 0, 136, 1, 3.88]
    ]
}

#print(len(data['features']))
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data['features'])

for i in range(len(scaled_features)):
    features = np.array(scaled_features[i]).reshape(1, -1)
    pred_next_1hr = models['model_1'].predict(features)
    print("Next 1hr: ", pred_next_1hr)
    # Append prediction and predict next_2hr
    features_next_2hr = np.column_stack([features, pred_next_1hr])
    pred_next_2hr = models['model_2'].predict(features_next_2hr)
    print("Next 2hr: ", pred_next_2hr)
    # Append prediction and predict next_3hr
    features_next_3hr = np.column_stack([features_next_2hr, pred_next_2hr])
    pred_next_3hr = models['model_3'].predict(features_next_3hr)
    print("Next 3hr: ", pred_next_3hr)
    # Append prediction and predict next_4hr
    features_next_4hr = np.column_stack([features_next_3hr, pred_next_3hr])
    pred_next_4hr = models['model_4'].predict(features_next_4hr)
    print("Next 4hr: ", pred_next_4hr) 

