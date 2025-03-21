import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


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
feature_names =data.columns[1:-1]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 8))
plt.rcParams['font.sans-serif']='kaiti'
barplot = sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df,
    palette='viridis',
    edgecolor='black'
)

plt.title('Feature importance', fontsize=25, fontweight='bold', pad=20)
plt.xlabel('Importance score', fontsize=25)
plt.ylabel('Feature', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

for index, value in enumerate(importance_df['Importance']):
    barplot.text(
        value + 0.005,
        index,
        f'{value:.4f}',
        va='center',
        fontsize=25
    )

plt.tight_layout()
plt.xlim(0, importance_df['Importance'].max() * 1.15)
plt.savefig('1.png', dpi=450, bbox_inches='tight')
plt.show()