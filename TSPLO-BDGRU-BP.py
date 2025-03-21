import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import math
from scipy.stats.qmc import Sobol
from sklearn.ensemble import RandomForestRegressor
import torch.nn.functional as F

def Levy(d):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step
def initon(N, dim, lb, ub):
    sobol_sampler = Sobol(d=dim, scramble=True)
    sobol_points = sobol_sampler.random(N)
    positions = lb + (ub - lb) * sobol_points
    return positions

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



def k_fold_cross_validation(X, y, model, criterion, optimizer, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_errors = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(
            1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_predictions = model(X_val_tensor).numpy()

        val_mse = mean_squared_error(y_val, val_predictions)
        val_errors.append(val_mse)

    return np.mean(val_errors)


def fobj(params):
    gru_hidden_dim, mlp_hidden_dim, lr, epochs, dropout_prob, weight = params

    model = GRU_BP_Model(input_dim=X_train_selected.shape[1],
                         gru_hidden_dim=int(gru_hidden_dim),
                         mlp_hidden_dim=int(mlp_hidden_dim),
                         output_dim=1,
                         dropout_prob=dropout_prob)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight)

    val_mse = k_fold_cross_validation(X_train_selected, y_train, model, criterion, optimizer, k=3)

    return val_mse


def chaos(ik):
    r = 2 - ik
    x = np.random.rand()
    if x < 1 / r:
        x = r * x
    else:
        x = (1 - x) / (1 - 1 / r)
    return x


def initon1(SearchAgents_no, dim, ub, lb):
    ub = np.array(ub)
    lb = np.array(lb)
    Boundary_no = ub.size

    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        assert Boundary_no == dim
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions

def TSPLO(N, MaxFEs, lb, ub, dim, Maiter, fobj):
    start_time = time.time()
    FEs = 0
    it = 0
    fitness = np.inf * np.ones(N)
    fitnew = np.inf * np.ones(N)
    X = initon(N, dim, lb, ub)
    V = np.ones((N, dim))
    X_new = np.zeros((N, dim))

    for i in range(N):
        fitness[i] = fobj(X[i, :])
        FEs += 1

    fitness, SortOrder = np.sort(fitness), np.argsort(fitness)
    X = X[SortOrder, :]
    Bestpos = X[0, :]
    Bests = fitness[0]
    Ccurve = np.zeros(int(MaxFEs / dim))

    while it < Maiter:
        X_sum = np.sum(X, axis=0)
        X_mean = X_sum / N
        w1 = 1 / (1 + np.exp(-((FEs / MaxFEs) ** 4)))
        w2 = np.exp(-(2 * FEs / MaxFEs) ** 3)
        ik = it / Maiter
        for i in range(N):
            a = np.random.rand() / 2 + 1
            V[i, :] = np.exp((1 - a) / 100 * FEs)
            LS = V[i, :]
            GS = chaos(ik) * Levy(dim) * (X_mean - X[i, :]) + lb / 2 + np.random.rand(1, dim) * (ub - lb) / 2
            X_new[i, :] = X[i, :] + (w1 * LS + w2 * GS) * np.random.rand(1, dim)
        for i in range(N):
            E = np.sqrt(FEs / MaxFEs)
            for j in range(dim):
                if np.random.rand() < 0.05 and np.random.rand() < E:
                    A_neighbour = np.random.permutation(N)[:2]
                    X_new[i, j] = X[i, j] + np.sin(np.random.rand() * np.pi) * (X[i, j] - X[A_neighbour[0], j]) + \
                                  np.cos(np.random.rand() * np.pi) * (X[i, j] - X[A_neighbour[1], j])

            Flag4ub = X_new[i, :] > ub
            Flag4lb = X_new[i, :] < lb
            X_new[i, :] = np.where(Flag4ub + Flag4lb, ub * Flag4ub + lb * Flag4lb, X_new[i, :])

            fitnew[i] = fobj(X_new[i, :])
            FEs += 1
            if fitnew[i] < fitness[i]:
                X[i, :] = X_new[i, :]
                fitness[i] = fitnew[i]

        fitness, SortOrder = np.sort(fitness), np.argsort(fitness)
        X = X[SortOrder, :]

        if fitness[0] < Bests:
            Bestpos = X[0, :]
            Bests = fitness[0]

        Ccurve[it] = Bests
        it += 1
    elapsed_time = time.time() - start_time
    return Bestpos, Bests, Ccurve, elapsed_time

def evaluate_model(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    mape = np.mean(np.abs((true_values - predictions) / true_values))*100
    mae = np.mean(np.abs(true_values - predictions))
    return mse, r2, mape, mae


data = pd.read_csv('Data.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
y_all = np.concatenate((y_train, y_test))
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
X_train_selected = X_train
X_test_selected = X_test

X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

N = 32
lb = np.array([16, 16, 1e-5, 1000, 0.01, 1e-5])
ub = np.array([256, 512, 0.01, 3000, 0.5, 0.01])
dim = 6
Maiter = 100
MaxFEs = Maiter * dim
Bestpos, Bests, Ccurve, elapsed_time = PLOrf(N, MaxFEs, lb, ub, dim, Maiter, fobj)
print(f"Best hyperparameters: {Bestpos}")
print(f"Time elapsed: {elapsed_time} seconds")

gru_hidden_dim, mlp_hidden_dim, lr, epochs, dropout_prob, weight = Bestpos
modelgbk = GRU_BP_Model(input_dim=X_train_selected.shape[1],
                        gru_hidden_dim=int(gru_hidden_dim),
                        mlp_hidden_dim=int(mlp_hidden_dim),
                        output_dim=1,
                        dropout_prob=dropout_prob)

criterion = nn.MSELoss()
optimizer = optim.Adam(modelgbk.parameters(), lr=lr, weight_decay=weight)
for epoch in range(int(epochs)):
    modelgbk.train()
    optimizer.zero_grad()
    outputs = modelgbk(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

modelgbk.eval()
with torch.no_grad():
    train_predictions = modelgbk(X_train_tensor).numpy()
    test_predictions = modelgbk(X_test_tensor).numpy()
predictions_allgbk = np.concatenate((train_predictions, test_predictions))
overall_mse, overall_r2, overall_mape, overall_mae = evaluate_model(y_all, np.squeeze(predictions_allgbk))
test_mse, test_r2, test_mape, test_mae = evaluate_model(y_test, np.squeeze(test_predictions))
train_mse, train_r2, train_mape, train_mae = evaluate_model(y_train, np.squeeze(train_predictions))
print(f"TSPLO-BDGRU-BP Overall MSE: {overall_mse:.4f}, R²: {overall_r2:.4f}, MAPE: {overall_mape:.4f}%, MAE: {overall_mae:.4f}")
print(f"Training Set MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAPE: {train_mape:.4f}%, MAE: {train_mae:.4f}")
print(f"Test Set MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAPE: {test_mape:.4f}%,MAE: {test_mae:.4f}")



