import os
import xarray as xr
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import numpy as np
import cmaps
import warnings
warnings.filterwarnings('ignore')
# matplotlib.use("Agg")

# setting aimed colormap for plotting
colorbar_plt = cmaps.WhiteBlueGreenYellowRed
colorbar_cmocean = cmaps.cmocean_deep


file_dir = r'C:/Users/87371/Downloads/'
file_name = 'NLDAS_MOS0125_M.A202001.020.nc'
dataset = xr.open_dataset(file_dir + file_name)
print(dataset)
variable_name = 'Streamflow'
Streamflow = dataset[variable_name]
print(Streamflow)
lon = dataset['lon']
lat = dataset['lat']

south_lat = 25
north_lat = 52
east_lon = -75
west_lon = -115
import geopandas as gpd

# Read the shapefile
shapefile_path = "C:/Users/87371/Downloads/MississippiRive/Miss_RiverBasin/reprojected_Miss_RiverBasin.shp"  # Replace with your shapefile path
gdf = gpd.read_file(shapefile_path)

# Print the attribute data (columns)
print(gdf.columns)

# Print the first few rows of the data
print(gdf.head())

map = Basemap(projection='cyl', resolution='l',
              llcrnrlat=south_lat, urcrnrlat=north_lat,
              llcrnrlon=west_lon, urcrnrlon=east_lon)
# C:\Users\87371\Downloads\MississippiRive\Miss_RiverBasin
map.readshapefile(r"C:/Users/87371/Downloads/MississippiRive/Miss_RiverBasin/reprojected_Miss_RiverBasin",
                  '1', drawbounds=True,
                  linewidth=0.2)

plt.pcolor(lon, lat, Streamflow[0,:,:], cmap=colorbar_plt)
plt.colorbar()
plt.show()
'''
for model_name in model_name_list:
    print(model_name)
    model_name_index = model_name_list.index(model_name)

    cm = 1 / 2.54
    dpi = 20
    fig = plt.figure(figsize=(15 * cm, 15 * cm), dpi=800)
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 7
    order_subplot = [0, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                     't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    for experiment_name in experiment_name_list:
        #print(experiment_name)
        experiment_name_index = experiment_name_list.index(experiment_name)
        ts_file_name_list_future_projection_subset = [file_name for file_name in ts_file_name_list_future_projection if
                                                      experiment_name in file_name and model_name in file_name]
        future_projection_file_url = file_dir + ts_file_name_list_future_projection_subset[0]
        future_projection_dataset = xr.open_dataset(future_projection_file_url)
        #print(future_projection_dataset)
        time_start = future_projection_dataset['time_start']
        time_end = future_projection_dataset['time_end']
        time_peak = future_projection_dataset['time_peak']
        time_start_max = time_start.max(dim=['lat', 'lon'], skipna=True)
        time_end_max = time_end.max(dim=['lat', 'lon'], skipna=True)
        time_peak_max = time_peak.max(dim=['lat','lon'], skipna=True)
        # print(time_start_max, time_end_max, time_peak_max)


        #print(time_start.min(dim=['lat','lon']).values, time_end, time_peak.shape)
        future_projection_variable = future_projection_dataset[variable_name]
        lon = future_projection_dataset['lon']
        lat = future_projection_dataset['lat']


            ax = plt.subplot(3, 4, time_id * 4 + experiment_name_index + 1)
            ax.spines['bottom'].set_linewidth(1 * dpi / 72)
            ax.spines['top'].set_linewidth(1 * dpi / 72)
            ax.spines['left'].set_linewidth(1 * dpi / 72)
            ax.spines['right'].set_linewidth(1 * dpi / 72)

            map = Basemap(projection='cyl', resolution='l',
                          llcrnrlat=south_lat, urcrnrlat=north_lat,
                          llcrnrlon=west_lon, urcrnrlon=east_lon)
            map.readshapefile(r"C:/Users/87371/Downloads/Mississippi__River_Valley_Sub-Basins/Mississippi__River_Valley_Sub-Basins", '1', drawbounds=True,
                              linewidth=0.2)

            plt.rc('font', family='Times New Roman')
            parallels = np.arange(-90, 90, 10.)
            # attribute labels = [left,right,top,bottom]
            if experiment_name_index != 0:
                map.drawparallels(parallels, labels=[False, False, False, True], fontsize=7, zorder=0,
                                  fontproperties='Times New Roman', linewidth=1 * dpi / 72)
            else:
                map.drawparallels(parallels, labels=[True, False, False, True], fontsize=7, zorder=0,
                                  fontproperties='Times New Roman', linewidth=1 * dpi / 72)
            meridians = np.arange(-180, 180, 10.)

            maps = map.drawmeridians(meridians, labels=[True, False, False, True], fontsize=7, zorder=0,
                                     fontproperties='Times New Roman', linewidth=1 * dpi / 72)

            ax1 = ax.pcolormesh(lon, lat, future_projection_variable_subset_mean, vmin=40.0,
                                vmax=100.0, cmap=colorbar_plt)

            sub_title = '$\mathrm{(}$' + '$\mathrm{0}$'.format(
                str('{') + order_subplot[(time_id * 4 + experiment_name_index + 1)] + str(
                    '}')) + '$\mathrm{)}$' + '_' + experiment_name + ' ' + '$\mathrm{0}$'.format(
                str('{') + str(start_date[0:4]) + str('}')) + '-' + '$\mathrm{0}$'.format(
                str('{') + str(end_date[0:4]) + str('}'))
            # print(sub_title)
            ax.text(0.50, 1.08, sub_title, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=6, fontproperties='Times New Roman', math_fontfamily='stix')
        del future_projection_variable_subset_mean

    cbar_ax = fig.add_axes([0.93, 0.35, 0.012, 0.30])
    cb = plt.colorbar(ax1, cax=cbar_ax)
    cb.outline.set_linewidth(1.5 * dpi / 72)
    cb.ax.tick_params(labelsize=6.5, which='both', direction='in', length=1, width=0.5, color='black', pad=1.5)
    lebal_title = '$\mathrm{0}$'.format(str('{') + 'Marine Heat Wave. duration' + str(
        '}')) + ' ' + '$\mathrm{(}$' + '$\mathrm{day·} \mathrm{time}^{-1}$' + '$\mathrm{)}$'
    # print(lebal_title)
    cb.ax.tick_params(size=0)
    cb.set_ticks(np.arange(40, 0.1 + 100.00001, 20))
    cb.set_label(lebal_title, loc='center', fontsize=6.5, rotation=270, fontproperties='Times New Roman',
                 math_fontfamily='stix', labelpad=6.5)
    plt.suptitle(model_name)
    plt.tight_layout(h_pad=0.3, w_pad=0.05, rect=[0, 0.01, 0.92, 0.95])
    plt.savefig('{}.jpg'.format('CMIP6_decadal_marine_heat_wave_{}_time'.format(variable_name) + model_name),
                dpi=800)
    del ax1
    plt.clf()

'''



