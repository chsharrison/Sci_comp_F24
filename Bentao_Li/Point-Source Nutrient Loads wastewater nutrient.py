import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmaps
import warnings
warnings.filterwarnings('ignore')


# setting aimed colormap for plotting
colorbar_plt = cmaps.WhiteBlueGreenYellowRed
colorbar_cmocean = cmaps.cmocean_deep
color_bugn = cmaps.MPL_BuGn

# Read the CSV file containing station data
csv_file = "C:/Users/87371/Downloads/Point_SourceNut/Facilities_CONUS.csv"
df = pd.read_csv(csv_file)

# Display the first few rows
print(df.shape)

# Create a GeoDataFrame from the CSV data
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf_stations = gpd.GeoDataFrame(df, geometry=geometry)

# Set the CRS (Coordinate Reference System) to WGS84 (EPSG:4326)
gdf_stations.set_crs(epsg=4326, inplace=True)

print(gdf_stations.shape)

#Point-Source Nutrient Loads

# Load the shapefile containing the boundary
shapefile_path = "C:/Users/87371/Downloads/MississippiRive/Miss_RiverBasin/reprojected_Miss_RiverBasin.shp"
gdf_boundary = gpd.read_file(shapefile_path)

# Check the CRS of the shapefile
print(gdf_boundary.crs)


# Ensure the CRS matches between the station points and the shapefile
if gdf_boundary.crs != gdf_stations.crs:
    gdf_boundary = gdf_boundary.to_crs(gdf_stations.crs)

# Perform spatial join: Check which stations are within the boundary
stations_within_boundary = gpd.sjoin(gdf_stations, gdf_boundary, how="inner", predicate="within")

# Save or display the results
print(stations_within_boundary)
stations_within_boundary.to_csv("stations_within_boundary.csv", index=False)

npdes_list = stations_within_boundary['npdes'].tolist()

stations_within_boundary_subset = stations_within_boundary[['npdes', 'Longitude', 'Latitude']]

# C:\Users\87371\Downloads\Point_SourceNut\TNPoutfall_moloads

# Read the CSV file containing station data
TN_file = "C:/Users/87371/Downloads/Point_SourceNut/TNP_yrloads/TNyrloads.csv"
TP_file = "C:/Users/87371/Downloads/Point_SourceNut/TNP_yrloads/TPyrloads.csv"
TN_dataset = pd.read_csv(TN_file)
TP_dataset = pd.read_csv(TP_file)
TN_dataset_subset = TN_dataset[TN_dataset['npdes'].isin(npdes_list)]
TP_dataset_subset = TP_dataset[TP_dataset['npdes'].isin(npdes_list)]
print(TP_dataset_subset)

TN_dataset_subset['year'] = pd.to_numeric(TN_dataset_subset['year'], errors='coerce')
TP_dataset_subset['year'] = pd.to_numeric(TP_dataset_subset['year'], errors='coerce')
TN_dataset_subset['loadTN'] = pd.to_numeric(TN_dataset_subset['loadTN'], errors='coerce')
TP_dataset_subset['loadTP'] = pd.to_numeric(TP_dataset_subset['loadTP'], errors='coerce')


cm = 1 / 2.54
dpi = 20
fig = plt.figure(figsize=(15 * cm, 13 * cm), dpi=800)
plt.rc('font', family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 7
order_subplot = [0, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



# Define table dimensions
row_name = [2000, 2005, 2010, 2015, 2020]
column_name = ['0-100', '100-1000', '1000-5000', '5000-', 'mean', 'sum']
# Create a blank DataFrame with NaN values
table = pd.DataFrame(index=row_name, columns=column_name)
print(table)
# Iterate over each year from 2000 to 2020
year_index = 0
for year in np.arange(2000, 2026, 5):
    if year_index <= 4:
        TN_dataset_subset_year = TN_dataset_subset[TN_dataset_subset['year'] == year]
        TP_dataset_subset_year = TP_dataset_subset[TP_dataset_subset['year'] == year]
        TN_dataset_subset_year_loc = pd.merge(TN_dataset_subset_year, stations_within_boundary_subset, on="npdes", how="left")
        TP_dataset_subset_year_loc = pd.merge(TP_dataset_subset_year, stations_within_boundary_subset, on="npdes", how="left")
        print(TN_dataset_subset_year.shape, TP_dataset_subset_year.shape)

        ranges = [(0, 100), (100, 1000), (1000, 5000), (5000, 500000000)]
        for lower, upper in ranges:
            range_index = ranges.index((lower, upper))
            # Filter data within the current range
            in_range = TN_dataset_subset_year_loc[(TN_dataset_subset_year_loc['loadTN'] > lower) & (TN_dataset_subset_year_loc['loadTN'] <= upper)]
            mean = np.mean(TN_dataset_subset_year_loc['loadTN'])
            sum = np.sum(TN_dataset_subset_year_loc['loadTN'])
            quantile_25 = TN_dataset_subset_year_loc['loadTN'].quantile(0.25)
            quantile_75 = TN_dataset_subset_year_loc['loadTN'].quantile(0.75)

            print(mean, quantile_25,quantile_75 )
            # Calculate statistics
            count = in_range.shape[0]
            # print(count)
            table.loc[year][range_index] = count
            table.loc[year]['mean'] = '%1.2f' % mean
            table.loc[year]['sum'] = '%1.2f' % sum
            # print(table)

        ax = plt.subplot(3, 2, year_index + 1)

        print(year_index)
        ax.spines['bottom'].set_linewidth(1 * dpi / 72)
        ax.spines['top'].set_linewidth(1 * dpi / 72)
        ax.spines['left'].set_linewidth(1 * dpi / 72)
        ax.spines['right'].set_linewidth(1 * dpi / 72)
        vmin, vmax = 0, 20000  # Custom value range
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        TN_dataset_subset_year_loc = TN_dataset_subset_year_loc.sort_values(by='loadTN', ascending=True)
        ax1 = plt.scatter(TN_dataset_subset_year_loc['Longitude'], TN_dataset_subset_year_loc['Latitude'], c = TN_dataset_subset_year_loc['loadTN'], s=1, norm = norm, cmap=colorbar_plt)
        sub_title = '$\mathrm{(}$' + '$\mathrm{0}$'.format(
            str('{') + order_subplot[(year_index + 1)] + str(
                '}')) + '$\mathrm{)}$' + ' ' + str(year)
        ax.text(0.50, 1.08, sub_title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
        year_index += 1
    else:
        ax = plt.subplot(3, 2, year_index + 1)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        print(table)

        table_plot = ax.table(cellText=table.values, colLabels=table.columns,
                              rowLabels=table.index, colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25, 0.28],
                              bbox=[0.05, 0.1, 0.98, 0.8], loc="center", edges='horizontal', fontsize=10)

        table_plot.auto_set_column_width(range(len(table.columns) + 1))

        sub_title = '$\mathrm{(}$' + '$\mathrm{0}$'.format(
            str('{') + order_subplot[(year_index + 1)] + str(
                '}')) + '$\mathrm{)}$' + ' ' + 'Statistics 2000-2020'
        ax.text(0.50, 0.95, sub_title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')

cbar_ax = fig.add_axes([0.93, 0.35, 0.012, 0.30])
cb = plt.colorbar(ax1, cax=cbar_ax)
cb.outline.set_linewidth(1.5 * dpi / 72)
cb.ax.tick_params(labelsize=6.5, which='both', direction='in', length=1, width=0.5, color='black', pad=1.5)
lebal_title = '$\mathrm{0}$'.format(str('{') + 'Nitrogen load' + str(
    '}')) + ' ' + '$\mathrm{(}$' + '$\mathrm{kgN·} \mathrm{yr}^{-1}$' + '$\mathrm{)}$'
# print(lebal_title)
cb.ax.tick_params(size=0)
cb.set_ticks(np.arange(vmin, 0.1 + vmax, 5000))
cb.set_label(lebal_title, loc='center', fontsize=6.5, rotation=270, fontproperties='Times New Roman',
             math_fontfamily='stix', labelpad=6.5)
plt.suptitle('Point-Source Nitrogen Loads to Streams' + ' ' + '$\mathrm{(}$' + '$\mathrm{kgN·} \mathrm{yr}^{-1}$' + '$\mathrm{)}$', fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
plt.tight_layout(h_pad=0.3, w_pad=0.4, rect=[0, 0.01, 0.92, 0.99])
plt.savefig('{}.jpg'.format('Point-Source Nitrogen wastewater'),
            dpi=800)




cm = 1 / 2.54
dpi = 20
fig = plt.figure(figsize=(15 * cm, 13 * cm), dpi=800)
plt.rc('font', family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 7
order_subplot = [0, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



# Define table dimensions
row_name = [2000, 2005, 2010, 2015, 2020]
column_name = ['0-100', '100-1000', '1000-5000', '5000-', 'mean', 'sum']
# Create a blank DataFrame with NaN values
table = pd.DataFrame(index=row_name, columns=column_name)
print(table)
# Iterate over each year from 2000 to 2020
year_index = 0
for year in np.arange(2000, 2026, 5):
    if year_index <= 4:
        TN_dataset_subset_year = TN_dataset_subset[TN_dataset_subset['year'] == year]
        TP_dataset_subset_year = TP_dataset_subset[TP_dataset_subset['year'] == year]
        TN_dataset_subset_year_loc = pd.merge(TN_dataset_subset_year, stations_within_boundary_subset, on="npdes", how="left")
        TP_dataset_subset_year_loc = pd.merge(TP_dataset_subset_year, stations_within_boundary_subset, on="npdes", how="left")
        print(TN_dataset_subset_year.shape, TP_dataset_subset_year.shape)

        ranges = [(0, 100), (100, 1000), (1000, 5000), (5000, 500000000)]
        for lower, upper in ranges:
            range_index = ranges.index((lower, upper))
            # Filter data within the current range
            in_range = TP_dataset_subset_year_loc[(TP_dataset_subset_year_loc['loadTP'] > lower) & (TP_dataset_subset_year_loc['loadTP'] <= upper)]
            mean = np.mean(TP_dataset_subset_year_loc['loadTP'])
            sum = np.sum(TP_dataset_subset_year_loc['loadTP'])
            quantile_25 = TP_dataset_subset_year_loc['loadTP'].quantile(0.25)
            quantile_75 = TP_dataset_subset_year_loc['loadTP'].quantile(0.75)

            print(mean, quantile_25,quantile_75 )
            # Calculate statistics
            count = in_range.shape[0]
            # print(count)
            table.loc[year][range_index] = count
            table.loc[year]['mean'] = '%1.2f' % mean
            table.loc[year]['sum'] = '%1.2f' % sum
            # print(table)

        ax = plt.subplot(3, 2, year_index + 1)

        print(year_index)
        ax.spines['bottom'].set_linewidth(1 * dpi / 72)
        ax.spines['top'].set_linewidth(1 * dpi / 72)
        ax.spines['left'].set_linewidth(1 * dpi / 72)
        ax.spines['right'].set_linewidth(1 * dpi / 72)
        vmin, vmax = 0, 5000  # Custom value range
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        TP_dataset_subset_year_loc = TP_dataset_subset_year_loc.sort_values(by='loadTP', ascending=True)
        ax1 = plt.scatter(TP_dataset_subset_year_loc['Longitude'], TP_dataset_subset_year_loc['Latitude'], c = TP_dataset_subset_year_loc['loadTP'], s=1, norm = norm, cmap=colorbar_plt)
        sub_title = '$\mathrm{(}$' + '$\mathrm{0}$'.format(
            str('{') + order_subplot[(year_index + 1)] + str(
                '}')) + '$\mathrm{)}$' + ' ' + str(year)
        ax.text(0.50, 1.08, sub_title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
        year_index += 1
    else:
        ax = plt.subplot(3, 2, year_index + 1)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        print(table)

        table_plot = ax.table(cellText=table.values, colLabels=table.columns,
                              rowLabels=table.index, colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25, 0.28],
                              bbox=[0.05, 0.1, 0.98, 0.8], loc="center", edges='horizontal', fontsize=10)

        table_plot.auto_set_column_width(range(len(table.columns) + 1))

        sub_title = '$\mathrm{(}$' + '$\mathrm{0}$'.format(
            str('{') + order_subplot[(year_index + 1)] + str(
                '}')) + '$\mathrm{)}$' + ' ' + 'Statistics 2000-2020'
        ax.text(0.50, 0.95, sub_title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')

cbar_ax = fig.add_axes([0.93, 0.35, 0.012, 0.30])
cb = plt.colorbar(ax1, cax=cbar_ax)
cb.outline.set_linewidth(1.5 * dpi / 72)
cb.ax.tick_params(labelsize=6.5, which='both', direction='in', length=1, width=0.5, color='black', pad=1.5)
lebal_title = '$\mathrm{0}$'.format(str('{') + 'Phosphorous load' + str(
    '}')) + ' ' + '$\mathrm{(}$' + '$\mathrm{kgP·} \mathrm{yr}^{-1}$' + '$\mathrm{)}$'
# print(lebal_title)
cb.ax.tick_params(size=0)
cb.set_ticks(np.arange(vmin, 0.1 + vmax, 1000))
cb.set_label(lebal_title, loc='center', fontsize=6.5, rotation=270, fontproperties='Times New Roman',
             math_fontfamily='stix', labelpad=6.5)
plt.suptitle('Point-Source Phosphorous  Loads to Streams' + ' ' + '$\mathrm{(}$' + '$\mathrm{kgP·} \mathrm{yr}^{-1}$' + '$\mathrm{)}$', fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
plt.tight_layout(h_pad=0.3, w_pad=0.4, rect=[0, 0.01, 0.92, 0.99])
plt.savefig('{}.jpg'.format('Point-Source Phosphorous wastewater'),
            dpi=800)

