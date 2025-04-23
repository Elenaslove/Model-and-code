import warnings
import geopandas as gpd
import libpysal as ps
import matplotlib.pyplot as plt
import pandas as pd
from esda.moran import Moran, Moran_Local
import numpy as np
import torch
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from splot.esda import moran_scatterplot

warnings.filterwarnings("ignore")
seed = 30
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
gdf = gpd.read_file('Beijing.shp')
carbon_data = pd.read_csv('carbon_esdata.csv')
gdf = gdf.merge(carbon_data, on='PAC')  # Assuming the merge field is 'PAC'

# Create a mapping dictionary from Chinese to English district names
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

# Apply K-means clustering, assuming we want to create 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
gdf['Cluster'] = kmeans.fit_predict(X)

# Construct Queen spatial weights matrix
w = ps.weights.Queen.from_dataframe(gdf)
w.transform = 'r'  # Row-standardize the weights

# Calculate global Moran's I
moran = Moran(gdf['2017'], w)
print(f"Global Moran's I: {moran.I:.3f}, p-value: {moran.p_sim:.3f}")

# Local Moran's I calculation and scatter plot
ga_moran_loc = Moran_Local(gdf['2017'], w, permutations=200)
fig2, ax0 = moran_scatterplot(ga_moran_loc, zstandard=False, p=0.05)

# Labelling the scatter plot
ax0.set_title("")  # Remove title
ax0.set_xlabel('Pct Bach')
ax0.set_ylabel('Spatial Lag of Pct Bach')
plt.show()

# Calculate LISA (Local Indicators of Spatial Association)
lisa = Moran_Local(gdf['2017'], w)

# Store clustering type and significance results in GeoDataFrame
gdf['LISA_Type'] = lisa.q  # Clustering type (1=HH, 2=LH, 3=LL, 4=HL)
gdf['LISA_Sig'] = lisa.p_sim < 0.05  # Significance flag

# Define labels for clustering types
lisa_labels = {
    1: 'HH (High-High)',
    2: 'LH (Low-High)',
    3: 'LL (Low-Low)',
    4: 'HL (High-Low)',
    0: 'NS (Not Significant)'  # Non-significant regions
}

# Assign clustering types to significant regions and 0 (non-significant) to others
gdf['LISA_Cluster'] = gdf['LISA_Type'].where(gdf['LISA_Sig'], 0)

# Visualization of clustering results
fig, ax1 = plt.subplots(figsize=(10, 10))
# Set color map
cmap = plt.get_cmap('tab10', 3)  # 'tab10' is a colormap with many colors, '3' specifies we need 3 colors
gdf.plot(column='Cluster', ax=ax1, cmap=cmap, legend=False)  # Don't use legend_kwds for colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2))
sm._A = []
fig.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.02, pad=0.02)

# Create legend manually
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
colors = [cmap(0), cmap(1), cmap(2)]  # Get the colors from the colormap
# Create legend items for each cluster
handles = [Patch(color=color, label=label) for color, label in zip(colors, legend_labels)]

# Annotate each district with its name
for idx, row in gdf.iterrows():
    # Get the centroid of each district
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    # Annotate the district's name at the centroid
    ax1.annotate(
        row['NAME_x'],  # District name field
        xy=(x, y),  # Location of the annotation
        xytext=(0, 0),  # Offset of the text
        textcoords="offset points",  # Offset unit is points
        ha='center',  # Horizontal alignment
        fontsize=12,  # Font size
        color='black',  # Font color
        alpha = 0.7,  # Text transparency
        rotation = -45  # Rotate text by -45 degrees
    )

# Adjust legend display, place it beside and shrink the size
ax1.legend(handles=handles)
plt.tight_layout()
# Save the first image
plt.savefig('figure_5.pdf', dpi=300, bbox_inches='tight')
# Clear the current figure
plt.clf()

# Plot for LISA Cluster Types
fig, ax1 = plt.subplots(figsize=(10, 10))
# First plot all areas as light grey (non-significant background)
gdf.plot(color='lightgrey', ax=ax1, edgecolor='k', linewidth=0.5)
# Then overlay significant and non-significant region clusters
gdf.plot(column='LISA_Cluster',
         categorical=True,
         legend=True,
         ax=ax1,
         legend_kwds={
             'title': 'LISA Cluster Type',
             'loc': 'lower right',
             'labels': [lisa_labels[i] for i in sorted(gdf['LISA_Cluster'].unique())]
         })

# Annotate each district with its name
for idx, row in gdf.iterrows():
    # Get the centroid of each district
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    # Annotate the district's name at the centroid
    ax1.annotate(
        row['NAME_x'],  # District name field
        xy=(x, y),  # Location of the annotation
        xytext=(0, 0),  # Offset of the text
        textcoords="offset points",  # Offset unit is points
        ha='center',  # Horizontal alignment
        fontsize=12,  # Font size
        color='black',  # Font color
        alpha=0.7,  # Text transparency
        rotation = -45  # Rotate text by -45 degrees
    )

# Adjust layout to remove empty spaces
plt.tight_layout()
# Save the second image
plt.savefig('figure_4.pdf', dpi=300, bbox_inches='tight')

# Reload data
gdf = gpd.read_file('Beijing.shp')
carbon_data = pd.read_csv('carbon_esdata.csv')
gdf = gdf.merge(carbon_data, on='PAC')  # Assuming the merge field is 'PAC'

# Construct Queen spatial weights matrix
w = ps.weights.Queen.from_dataframe(gdf)
w.transform = 'r'  # Row-standardize the weights

# Create a data table to store Moran's I, p-value, and Z-value for each year
moran_results = []

# Loop through each year to calculate Moran's I
for year in range(1997, 2018):
    # Calculate global Moran's I
    moran = Moran(gdf[str(year)], w)

    # Retrieve Moran's I, p-value, and Z-value
    moran_results.append({
        'Year': year,
        'Moran_I': moran.I,
        'p_value': moran.p_sim,
        'Z_value': moran.z_sim
    })

# Convert the results to a DataFrame
moran_df = pd.DataFrame(moran_results)

# Print out Moran's I, p-value, and Z-value for each year
print(moran_df)
