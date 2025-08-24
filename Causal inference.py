import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import inv
from causallearn.search.ScoreBased.GES import ges
from sklearn.ensemble import RandomForestRegressor

seed = 30
np.random.seed(seed)
data = pd.read_csv('Data.csv')
X = data.iloc[:, 1:-1].values  # 跳过第一列和最后一列
y = data.iloc[:, -1].values.reshape(-1, 1)  # y是目标变量
KSSS = y

data_matrix = np.hstack([X, y])
result = ges(data_matrix)
n_nodes = result['G'].get_num_nodes()
adj_matrix = np.zeros((n_nodes, n_nodes))
for edge in result['G'].get_graph_edges():
    i = int(edge.get_node1().get_name()[1:]) - 1  # 转换x1→0, x2→1等
    j = int(edge.get_node2().get_name()[1:]) - 1
    adj_matrix[i, j] = 1
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
G = nx.DiGraph()
custom_names = {
    0: "x$_{1}$",  # Per capita disposable income (yuan)
    1: "x$_{2}$",  # Resident population (in ten thousand)
    2: "x$_{3}$",  # Passenger traffic (in ten thousand)
    3: "x$_{4}$",  # Public transport operating vehicles (vehicles)
    4: "x$_{5}$",  # Gross regional product (100 million yuan)
    5: "x$_{6}$",  # Consumer price index (1978=100)
    6: "x$_{7}$",  # Number of cars per 100 urban households (vehicles)
    7: "x$_{8}$",  # Government general public budget expenditure (100 million yuan)
    8: "x$_{9}$",  # Energy consumption total (10,000 tons of standard coal)
    9: "x$_{10}$",  # Internal expenditure on research and experimental development funds (10,000 yuan)
    10: "x$_{11}$",  # Annual emission in Beijing (in million tons)
    11: "x$_{0}$"  # (原x1已移除，所有变量重新编号)
}
for i, name in custom_names.items():
    G.add_node(i, label=name)
# 添加边
for edge in result['G'].get_graph_edges():
    src = edge.get_node1().get_name()
    dest = edge.get_node2().get_name()
    G.add_edge(int(src[1:]) - 1, int(dest[1:]) - 1)
pos = {
    0: (0.2, 0.8), 1: (0.7, 0.9), 2: (0.5, 0.6), 3: (0.9, 0.7),
    4: (0.3, 0.4), 5: (0.8, 0.3), 6: (0.6, 0.1), 7: (0.1, 0.5),
    8: (0.4, 0.2), 9: (0.9, 0.5), 10: (0.2, 0.1), 11: (0.5, 0.8)
}
plt.figure(figsize=(14, 12))
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='black', linewidths=2)
for idx, (x, y) in pos.items():
    plt.text(x, y, custom_names[idx], ha='center', va='center', bbox=dict(color='lightblue', edgecolor='none', pad=1), fontsize=12)
nx.draw_networkx_edges(G, pos, edge_color='black', width=3, arrows=True, arrowstyle='->', arrowsize=45, connectionstyle='arc3,rad=0.2')
plt.tight_layout()
plt.savefig('custom_named_causal_graph.png', dpi=300)
plt.show()
def estimate_tmle_effect(X, treatment, outcome, model=RandomForestRegressor()):
    treat_model = model.fit(X, treatment)
    treat_pred = treat_model.predict(X)
    outcome_model = model.fit(X, outcome)
    outcome_pred = outcome_model.predict(X)
    residuals_outcome = outcome - outcome_pred
    residuals_treatment = treatment - treat_pred
    tmle_effect = np.mean(residuals_outcome * residuals_treatment) / np.var(residuals_treatment)
    return tmle_effect
import networkx as nx
import numpy as np
def find_all_paths(graph, source, target, path=[]):
    path = path + [source]
    if source == target:
        return [path]
    if source not in graph:
        return []
    paths = []
    for neighbor in graph[source]:
        if neighbor not in path:
            new_paths = find_all_paths(graph, neighbor, target, path)
            for p in new_paths:
                paths.append(p)
    return paths


def compute_total_effect_with_tmle(adj_matrix, graph, source, target, X, outcome):
    direct_effect = adj_matrix[source, target]
    all_paths = find_all_paths(graph, source, target)

    # 计算路径数
    num_paths = len(all_paths)

    if num_paths == 0:  # 没有路径时直接返回直接效应
        return direct_effect

    # 初始化总效应为直接效应
    total_effect = direct_effect

    # 计算每条路径的效应
    path_effects = []
    for path in all_paths:
        path_effect = 1.0
        for i in range(len(path) - 1):
            edge_weight = adj_matrix[path[i], path[i + 1]]
            path_effect *= edge_weight

        treatment = X[:, path[0]]
        path_tmle_effect = estimate_tmle_effect(X, treatment, outcome)

        path_effects.append(path_tmle_effect * path_effect)

    # 平均路径效应
    avg_path_effect = np.mean(path_effects)

    total_effect += avg_path_effect  # 将平均路径效应加入总效应

    return total_effect
G = nx.DiGraph()
for i, name in custom_names.items():
    G.add_node(i, label=name)
for edge in result['G'].get_graph_edges():
    src = edge.get_node1().get_name()
    dest = edge.get_node2().get_name()
    G.add_edge(int(src[1:]) - 1, int(dest[1:]) - 1)
effects = {}
for src in range(len(custom_names) - 1):
    treatment = X[:, src]
    outcome = KSSS
    effect = compute_total_effect_with_tmle(adj_matrix, G, src, 11,X, outcome)
    effects[custom_names[src]] = effect
print("\n=== 各特征对碳排放量的总因果效应 ===")
for feature, effect in effects.items():
    print(f"{feature} → 碳排放量: 总效应={effect:.3f}")
# 可视化所有因果效应的条形图
plt.figure(figsize=(10, 8))
features = list(effects.keys())
values = list(effects.values())
plt.barh(features, values, color='lightcoral')
plt.xlabel("ATE causal effect")
plt.tight_layout()
plt.show()