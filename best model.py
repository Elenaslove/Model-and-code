import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn.functional as F


class GRU_BP_Model(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, mlp_hidden_dim, output_dim, dropout_prob, num_gru_layers=2):
        super(GRU_BP_Model, self).__init__()
        self.gru = nn.GRU(input_dim, gru_hidden_dim, num_layers=num_gru_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(gru_hidden_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm1 = nn.BatchNorm1d(mlp_hidden_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = F.silu(self.fc1(out))
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def evaluate_model(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    mae = np.mean(np.abs(true_values - predictions))
    return mse, r2, mape, mae

seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
data = pd.read_csv('Data.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
y_all = np.concatenate((y_train, y_test))
X_train_selected = X_train
X_test_selected = X_test
X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# bdgru-bp
gru_hidden_dim = 256
mlp_hidden_dim = 511
lr = 0.01
epochs = 3000
dropout_prob = 0.5
weight_decay = 0.01
modelgb = GRU_BP_Model(input_dim=X_train_selected.shape[1],
                       gru_hidden_dim=gru_hidden_dim,
                       mlp_hidden_dim=mlp_hidden_dim,
                       output_dim=1,
                       dropout_prob=dropout_prob)  # 加入 dropout_prob 参数

criterion = nn.MSELoss()
optimizer = optim.Adam(modelgb.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in range(epochs):
    modelgb.train()
    optimizer.zero_grad()
    outputs = modelgb(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

modelgb.eval()
with torch.no_grad():
    train_predictions = modelgb(X_train_tensor).numpy()
    test_predictions = modelgb(X_test_tensor).numpy()
predictions_allgb = np.concatenate((train_predictions, test_predictions))
overall_mse, overall_r2, overall_mape, overall_mae = evaluate_model(y_all, np.squeeze(predictions_allgb))
test_mse, test_r2, test_mape, test_mae = evaluate_model(y_test, np.squeeze(test_predictions))
train_mse, train_r2, train_mape, train_mae = evaluate_model(y_train, np.squeeze(train_predictions))
print(f"BDGRU-BP Overall MSE: {overall_mse:.4f}, R²: {overall_r2:.4f}, MAPE: {overall_mape:.4f}%, MAE: {overall_mae:.4f}")
print(f"Training Set MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAPE: {train_mape:.4f}%, MAE: {train_mae:.4f}")
print(f"Test Set MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAPE: {test_mape:.4f}%,MAE: {test_mae:.4f}")
