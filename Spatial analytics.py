import warnings
import geopandas as gpd
import libpysal as ps
import matplotlib.pyplot as plt
import pandas as pd
from esda.moran import Moran, Moran_Local
import numpy as np
import torch
from sklearn.cluster import KMeans
from adjustText import adjust_text
warnings.filterwarnings("ignore")
seed = 30
np.random.seed(seed)
torch.manual_seed(seed)
gdf = gpd.read_file('Beijing.shp')
carbon_data = pd.read_csv('carbon_esdata.csv')
gdf = gdf.merge(carbon_data, on='PAC')

district_name_map = {
    '朝阳区': 'Chaoyang',
    '通州区': 'Tongzhou',
    '门头沟区': 'Mentougou',
    '怀柔区': 'Huairou',
    '昌平区': 'Changping',
    '海淀区': 'Haidian',
    '平谷区': 'Pinggu',
    '顺义区': 'Shunyi',
    '石景山区': 'Shijingshan',
    '丰台区': 'Fengtai',
    '西城区': 'Xicheng',
    '东城区': 'Dongcheng',
    '大兴区': 'Daxing',
    '房山区': 'Fangshan'
}

gdf['NAME_x'] = gdf['NAME_x'].map(district_name_map)
X = gdf[['2017']].values
kmeans = KMeans(n_clusters=3, random_state=42)
gdf['Cluster'] = kmeans.fit_predict(X)

w = ps.weights.Queen.from_dataframe(gdf)
w.transform = 'r'
moran = Moran(gdf['2017'], w)
y = gdf['2017'].values
spatial_lag = ps.weights.lag_spatial(w, y)
moran_global = Moran(y, w)
moran_loc = Moran_Local(y, w, permutations=999)
significant = moran_loc.p_sim < 0.05
colors = np.where(significant, 'steelblue', 'lightgray')
edgecolors = np.where(significant, 'darkblue', 'gray')
plt.figure(figsize=(10, 8))
sc = plt.scatter(y, spatial_lag, c=colors, edgecolor=edgecolors,
                s=100, linewidth=1.2, alpha=0.8, zorder=5)
fit_coef = np.polyfit(y, spatial_lag, 1)
regression_line = fit_coef[0] * y + fit_coef[1]
plt.plot(y, regression_line,
         color='red', linewidth=2, linestyle='--',
         label=f"Regression slope = {fit_coef[0]:.2f}", zorder=3)
texts = []
for i in range(len(y)):
    if significant[i]:
        texts.append(plt.text(y[i], spatial_lag[i], gdf['NAME_x'].iloc[i],
                             fontsize=9, ha='center', va='center', color='black',
                             bbox=dict(boxstyle='round,pad=0.3',
                                     fc='white', alpha=0.8, ec='none'),
                             zorder=10))
    else:
        texts.append(plt.text(y[i], spatial_lag[i], gdf['NAME_x'].iloc[i],
                             fontsize=8, ha='center', va='center', color='grey',
                             alpha=0.7, zorder=9))
adjust_text(texts,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.6),
            expand_points=(1.2, 1.2))
plt.axhline(y=np.mean(spatial_lag), color='gray', linestyle=':', alpha=0.7)
plt.axvline(x=np.mean(y), color='gray', linestyle=':', alpha=0.7)
quadrant_labels = ['HH', 'LH', 'LL', 'HL']
quadrant_positions = [(0.9,0.9), (0.1,0.9), (0.1,0.1), (0.9,0.1)]
for label, pos in zip(quadrant_labels, quadrant_positions):
    plt.text(pos[0], pos[1], label, transform=plt.gca().transAxes,
            fontsize=12, weight='bold', alpha=0.6)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Significant',
              markerfacecolor='steelblue', markersize=10, markeredgecolor='darkblue'),
    plt.Line2D([0], [0], marker='o', color='w', label='Not Significant',
              markerfacecolor='lightgray', markersize=10, markeredgecolor='gray'),
]
plt.legend(handles=legend_elements, loc='upper right', framealpha=0.8)
plt.xlabel('Pct Bach', labelpad=10)
plt.ylabel('Spatial Lag of Pct Bach', labelpad=10)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('figure_3.pdf', dpi=300, bbox_inches='tight')
plt.show()

lisa = Moran_Local(gdf['2017'], w)
gdf['LISA_Type'] = lisa.q
gdf['LISA_Sig'] = lisa.p_sim < 0.05
lisa_labels = {
    1: 'HH (High-High)',
    2: 'LH (Low-High)',
    3: 'LL (Low-Low)',
    4: 'HL (High-Low)',
    0: 'NS (Not Significant)'  }
gdf['LISA_Cluster'] = gdf['LISA_Type'].where(gdf['LISA_Sig'], 0)

fig, ax1 = plt.subplots(figsize=(14, 12))
cmap = plt.get_cmap('viridis', 3)
gdf.plot(column='Cluster', ax=ax1, cmap=cmap, edgecolor='white', linewidth=0.8, legend=False)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2))
sm._A = []
cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.03, pad=0.02)
cbar.set_ticks([0.33, 1.0, 1.66])
cbar.set_ticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3'])
labeled_areas = {
    'Dongcheng': {'offset': (50, -20), 'direction': 'south'},
    'Xicheng': {'offset': (20, 20), 'direction': 'east'},
    'Chaoyang': {'offset': (0, 20), 'direction': 'north'},
}
def is_overlap(new_pos, existing_positions, min_distance=20):
    for pos in existing_positions:
        if ((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2) ** 0.5 < min_distance:
            return True
    return False
label_positions = []
for idx, row in gdf.iterrows():
    geom = row['geometry']
    name = row['NAME_x']
    area = geom.area
    if area < 0.02:
        x, y = geom.representative_point().x, geom.representative_point().y
    else:
        x, y = geom.centroid.x, geom.centroid.y
    fontsize = 10
    offset = (15, 10)
    rotation = 0
    bbox_props = dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='none')
    arrow_props = None
    text_pos = (x + offset[0], y + offset[1])
    if name in labeled_areas:
        config = labeled_areas[name]
        base_offset = config['offset']
        if config['direction'] == 'north':
            offset = (base_offset[0], abs(base_offset[1]))
        elif config['direction'] == 'south':
            offset = (base_offset[0], -abs(base_offset[1]))
        elif config['direction'] == 'west':
            offset = (-abs(base_offset[0]), base_offset[1])
        elif config['direction'] == 'northwest':
            offset = (-abs(base_offset[0]), abs(base_offset[1]))
        text_pos = (x + offset[0], y + offset[1])
        adjust_factor = 1
        while is_overlap(text_pos, label_positions):
            adjust_factor += 0.2
            offset = (base_offset[0] * adjust_factor, base_offset[1] * adjust_factor)
            text_pos = (x + offset[0], y + offset[1])
        arrow_props = dict(arrowstyle="->", color='red', lw=1.5, alpha=0.8, shrinkA=5)
    label_positions.append(text_pos)
    ax1.annotate(
        name,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        ha='center',
        fontsize=fontsize,
        color='black',
        bbox=bbox_props,
        rotation=rotation,
        arrowprops=arrow_props
    )

ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_5.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots(figsize=(14, 12))
gdf.plot(color='lightgrey', ax=ax1, edgecolor='k', linewidth=0.5)
gdf.plot(column='LISA_Cluster',
         categorical=True,
         legend=True,
         ax=ax1,
         legend_kwds={
             'title': 'LISA Cluster Type',
             'loc': 'lower right',
             'labels': [lisa_labels[i] for i in sorted(gdf['LISA_Cluster'].unique())]
         })

labeled_areas = {
    'Dongcheng': {'offset': (50, -20), 'direction': 'south'},
    'Xicheng': {'offset': (20, 20), 'direction': 'east'},
    'Chaoyang': {'offset': (0, 20), 'direction': 'north'},
}

def is_overlap(new_pos, existing_positions, min_distance=20):
    for pos in existing_positions:
        if ((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2) ** 0.5 < min_distance:
            return True
    return False
label_positions = []

for idx, row in gdf.iterrows():
    geom = row['geometry']
    name = row['NAME_x']
    area = geom.area
    if area < 0.02:
        x, y = geom.representative_point().x, geom.representative_point().y
    else:
        x, y = geom.centroid.x, geom.centroid.y

    fontsize = 10
    offset = (15, 10)
    rotation = 0
    bbox_props = dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='none')
    arrow_props = None
    text_pos = (x + offset[0], y + offset[1])

    if name in labeled_areas:
        config = labeled_areas[name]
        base_offset = config['offset']

        if config['direction'] == 'north':
            offset = (base_offset[0], abs(base_offset[1]))
        elif config['direction'] == 'south':
            offset = (base_offset[0], -abs(base_offset[1]))
        elif config['direction'] == 'west':
            offset = (-abs(base_offset[0]), base_offset[1])
        elif config['direction'] == 'northwest':
            offset = (-abs(base_offset[0]), abs(base_offset[1]))
        text_pos = (x + offset[0], y + offset[1])
        adjust_factor = 1
        while is_overlap(text_pos, label_positions):
            adjust_factor += 0.2
            offset = (base_offset[0] * adjust_factor, base_offset[1] * adjust_factor)
            text_pos = (x + offset[0], y + offset[1])

        arrow_props = dict(arrowstyle="->", color='red', lw=1.5, alpha=0.8, shrinkA=5)

    label_positions.append(text_pos)
    ax1.annotate(
        name,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        ha='center',
        fontsize=fontsize,
        color='black',
        bbox=bbox_props,
        rotation=rotation,
        arrowprops=arrow_props
    )

ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_4.pdf', dpi=300, bbox_inches='tight')
plt.show()
gdf = gpd.read_file('Beijing.shp')
carbon_data = pd.read_csv('carbon_esdata.csv')
gdf = gdf.merge(carbon_data, on='PAC')
w = ps.weights.Queen.from_dataframe(gdf)
w.transform = 'r'
moran_results = []
for year in range(1997, 2018):
    # Calculate global Moran's I
    moran = Moran(gdf[str(year)], w)
    moran_results.append({
        'Year': year,
        'Moran_I': moran.I,
        'p_value': moran.p_sim,
        'Z_value': moran.z_sim
    })
moran_df = pd.DataFrame(moran_results)
print(moran_df)