import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmaps
import warnings
warnings.filterwarnings('ignore')
# matplotlib.use("Agg")

# Define table dimensions
rows = 5
columns = 6
row_name = ['2000', '2005', '2010', '2015', '2020']
column_name = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-', 'mean', 'std dev']

# Create a blank DataFrame with NaN values
table = pd.DataFrame(index=row_name, columns=column_name)
print(table)
colorbar_plt = cmaps.WhiteBlueGreenYellowRed

# setting aimed colormap for plotting
colorbar_plt = cmaps.WhiteBlueGreenYellowRed
colorbar_cmocean = cmaps.cmocean_deep
color_bugn = cmaps.MPL_BuGn
shapefile_path = "C:/Users/87371/Downloads/MississippiRive/Miss_RiverBasin/reprojected_Miss_RiverBasin.shp"  # Replace with your shapefile path

# Step 2: Read the shapefile using GeoPandas
shapefile = gpd.read_file(shapefile_path)

year_list = ['2000', '2005', '2010', '2015', '2020', 'hist']

cm = 1 / 2.54
dpi = 20
fig = plt.figure(figsize=(15 * cm, 13 * cm), dpi=800)
plt.rc('font', family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 7
order_subplot = [0, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for year in year_list:
    if year != 'hist':
        print(year)
        year_index = year_list.index(year)
        file_dir = f'C:/Users/87371/Downloads/n_tw-{year}/'
        file_name  = f'n_tw-{year}.tif'
        with rasterio.open(file_dir + file_name) as src:
            raster = src.read(1)  # Read the first band
            raster_meta = src.meta
            transform = src.transform  # Affine transformation
            raster_crs = src.crs
            # Step 3: Define the target CRS (EPSG:4326 for longitude/latitude)
            target_crs = 'EPSG:4326'

            if src.crs != 'EPSG:4326':
                # Define the target CRS (EPSG:4326 for longitude/latitude)
                target_crs = 'EPSG:4326'

                # Step 3: Calculate the transform for the target CRS
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )

                # Create metadata for the new raster
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                # Step 4: Create an output raster to store the reprojected data
                with rasterio.open(file_dir + f'n_tw-{year}_reprojected.tif', 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest  # You can change the resampling method if needed
                        )

                # Open the reprojected raster
                with rasterio.open(file_dir + f'n_tw-{year}_reprojected.tif') as dst:
                    data = dst.read(1)
                    bounds = dst.bounds

            else:
                # If the raster is already in EPSG:4326, just read the data
                data = src.read(1)
                bounds = src.bounds

        # Step 3: Open the raster file using rasterio
        with rasterio.open(file_dir + f'n_tw-{year}_reprojected.tif') as src:
            # Step 4: Check if the CRS of the shapefile matches the raster's CRS
            if shapefile.crs != src.crs:
                shapefile = shapefile.to_crs(src.crs)  # Reproject shapefile to raster's CRS

            # Step 5: Mask the raster using the shapefile geometry
            # The shapefile geometry is passed as a list to the mask function
            geometry = shapefile.geometry.values  # Convert shapefile geometries to a list
            out_image, out_transform = mask(src, geometry, crop=True)  # Mask and crop the raster

            # Step 6: Update the metadata of the raster after masking
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "count": 1,
                "crs": src.crs,
                "transform": out_transform,
                "width": out_image.shape[2],
                "height": out_image.shape[1]
            })

        ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 500000)]
        for lower, upper in ranges:
            range_index = ranges.index((lower, upper))
            # Filter data within the current range
            in_range = data[(data > lower) & (data <= upper) & (data > 0.0)]
            mean = np.mean(data[data > 0])
            std_dev = np.std(data[data > 0])
            print(mean, std_dev)
            # Calculate statistics
            count = len(in_range)
            # print(count)
            table.loc[year][range_index] = count
            table.loc[year]['mean'] = '%1.2f' % mean
            table.loc[year]['std dev'] = '%1.2f' % std_dev
            print(table)
        vmin, vmax = 0, 25  # Custom value range
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # Normalization
        cmap = colorbar_plt # Colormap

        ax = plt.subplot(3, 2, year_index + 1)
        ax.spines['bottom'].set_linewidth(1 * dpi / 72)
        ax.spines['top'].set_linewidth(1 * dpi / 72)
        ax.spines['left'].set_linewidth(1 * dpi / 72)
        ax.spines['right'].set_linewidth(1 * dpi / 72)

        ax1 = ax.imshow(out_image[0], cmap=colorbar_plt, extent = [bounds[0], bounds[2], bounds[1], bounds[3]], norm= norm)  # Set extent to match lon/lat bounds
        sub_title = '$\mathrm{(}$' + '$\mathrm{0}$'.format(
            str('{') + order_subplot[(year_index + 1)] + str(
                '}')) + '$\mathrm{)}$' + ' ' + year
        ax.text(0.50, 1.08, sub_title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
    else:
        print(year)
        year_index = year_list.index(year)
        ax = plt.subplot(3, 2, year_index + 1)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        table_plot = ax.table(cellText=table.values, colLabels=table.columns,
                              rowLabels=table.index, colWidths=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                              bbox=[0.05, 0.1, 0.98, 0.8], loc="center", edges='horizontal',  fontsize=10)


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
lebal_title = '$\mathrm{0}$'.format(str('{') + 'Total (wet + dry) nitrogen deposition' + str(
    '}')) + ' ' + '$\mathrm{(}$' + '$\mathrm{kgN·} \mathrm{ha}^{-1}$' + '$\mathrm{)}$'
# print(lebal_title)
cb.ax.tick_params(size=0)
cb.set_ticks(np.arange(vmin, 0.1 + vmax, 5))
cb.set_label(lebal_title, loc='center', fontsize=6.5, rotation=270, fontproperties='Times New Roman',
             math_fontfamily='stix', labelpad=6.5)
plt.suptitle('NADP total nitrogen deposition' + ' ' + '$\mathrm{(}$' + '$\mathrm{kgN·} \mathrm{ha}^{-1}$' + '$\mathrm{)}$', fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
plt.tight_layout(h_pad=0.3, w_pad=0.4, rect=[0, 0.01, 0.92, 0.99])
plt.savefig('{}.jpg'.format('NADP total N deposition'),
            dpi=800)




# Define table dimensions
rows = 5
columns = 6
row_name = ['2000', '2005', '2010', '2015', '2020']
column_name = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-', 'mean', 'std dev']

# Create a blank DataFrame with NaN values
table = pd.DataFrame(index=row_name, columns=column_name)
print(table)

# setting aimed colormap for plotting
colorbar_plt = cmaps.WhiteBlueGreenYellowRed
colorbar_cmocean = cmaps.cmocean_deep
color_bugn = cmaps.MPL_BuGn
shapefile_path = "C:/Users/87371/Downloads/MississippiRive/Miss_RiverBasin/reprojected_Miss_RiverBasin.shp"  # Replace with your shapefile path

# Step 2: Read the shapefile using GeoPandas
shapefile = gpd.read_file(shapefile_path)

year_list = ['2000', '2005', '2010', '2015', '2020', 'hist']

cm = 1 / 2.54
dpi = 20
fig = plt.figure(figsize=(15 * cm, 13 * cm), dpi=800)
plt.rc('font', family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 7
order_subplot = [0, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for year in year_list:
    if year != 'hist':
        print(year)
        year_index = year_list.index(year)
        file_dir = f'C:/Users/87371/Downloads/s_tw-{year}/'
        file_name  = f's_tw-{year}.tif'
        with rasterio.open(file_dir + file_name) as src:
            raster = src.read(1)  # Read the first band
            raster_meta = src.meta
            transform = src.transform  # Affine transformation
            raster_crs = src.crs
            # Step 3: Define the target CRS (EPSG:4326 for longitude/latitude)
            target_crs = 'EPSG:4326'

            if src.crs != 'EPSG:4326':
                # Define the target CRS (EPSG:4326 for longitude/latitude)
                target_crs = 'EPSG:4326'

                # Step 3: Calculate the transform for the target CRS
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )

                # Create metadata for the new raster
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                # Step 4: Create an output raster to store the reprojected data
                with rasterio.open(file_dir + f's_tw-{year}_reprojected.tif', 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest  # You can change the resampling method if needed
                        )

                # Open the reprojected raster
                with rasterio.open(file_dir + f's_tw-{year}_reprojected.tif') as dst:
                    data = dst.read(1)
                    bounds = dst.bounds

            else:
                # If the raster is already in EPSG:4326, just read the data
                data = src.read(1)
                bounds = src.bounds

        # Step 3: Open the raster file using rasterio
        with rasterio.open(file_dir + f's_tw-{year}_reprojected.tif') as src:
            # Step 4: Check if the CRS of the shapefile matches the raster's CRS
            if shapefile.crs != src.crs:
                shapefile = shapefile.to_crs(src.crs)  # Reproject shapefile to raster's CRS

            # Step 5: Mask the raster using the shapefile geometry
            # The shapefile geometry is passed as a list to the mask function
            geometry = shapefile.geometry.values  # Convert shapefile geometries to a list
            out_image, out_transform = mask(src, geometry, crop=True)  # Mask and crop the raster

            # Step 6: Update the metadata of the raster after masking
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "count": 1,
                "crs": src.crs,
                "transform": out_transform,
                "width": out_image.shape[2],
                "height": out_image.shape[1]
            })

        ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 500000)]
        for lower, upper in ranges:
            range_index = ranges.index((lower, upper))
            # Filter data within the current range
            in_range = data[(data > lower) & (data <= upper) & (data > 0.0)]
            mean = np.mean(data[data > 0])
            std_dev = np.std(data[data > 0])
            print(mean, std_dev)
            # Calculate statistics
            count = len(in_range)
            # print(count)
            table.loc[year][range_index] = count
            table.loc[year]['mean'] = '%1.2f' % mean
            table.loc[year]['std dev'] = '%1.2f' % std_dev
            print(table)
        vmin, vmax = 0, 40  # Custom value range
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # Normalization
        cmap = colorbar_plt # Colormap

        ax = plt.subplot(3, 2, year_index + 1)
        ax.spines['bottom'].set_linewidth(1 * dpi / 72)
        ax.spines['top'].set_linewidth(1 * dpi / 72)
        ax.spines['left'].set_linewidth(1 * dpi / 72)
        ax.spines['right'].set_linewidth(1 * dpi / 72)

        ax1 = ax.imshow(out_image[0], cmap=colorbar_plt, extent = [bounds[0], bounds[2], bounds[1], bounds[3]], norm= norm)  # Set extent to match lon/lat bounds
        sub_title = '$\mathrm{(}$' + '$\mathrm{0}$'.format(
            str('{') + order_subplot[(year_index + 1)] + str(
                '}')) + '$\mathrm{)}$' + ' ' + year
        ax.text(0.50, 1.08, sub_title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
    else:
        print(year)
        year_index = year_list.index(year)
        ax = plt.subplot(3, 2, year_index + 1)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        table_plot = ax.table(cellText=table.values, colLabels=table.columns,
                              rowLabels=table.index, colWidths=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                              bbox=[0.05, 0.1, 0.98, 0.8], loc="center", edges='horizontal',  fontsize=10)


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
lebal_title = '$\mathrm{0}$'.format(str('{') + 'Total (wet + dry) sulfur deposition' + str(
    '}')) + ' ' + '$\mathrm{(}$' + '$\mathrm{kgS·} \mathrm{ha}^{-1}$' + '$\mathrm{)}$'
# print(lebal_title)
cb.ax.tick_params(size=0)
cb.set_ticks(np.arange(vmin, 0.1 + vmax, 10))
cb.set_label(lebal_title, loc='center', fontsize=6.5, rotation=270, fontproperties='Times New Roman',
             math_fontfamily='stix', labelpad=6.5)
plt.suptitle('NADP total sulfur deposition' + ' ' + '$\mathrm{(}$' + '$\mathrm{kgS·} \mathrm{ha}^{-1}$' + '$\mathrm{)}$', fontsize=9, fontproperties='Times New Roman', math_fontfamily='stix')
plt.tight_layout(h_pad=0.3, w_pad=0.4, rect=[0, 0.01, 0.92, 0.99])
plt.savefig('{}.jpg'.format('NADP total S deposition'),
            dpi=800)

