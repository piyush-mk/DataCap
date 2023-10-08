import os
import h5py
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import KMeans

def load_hyperion_data(data_path):
    all_files = [file for file in os.listdir(data_path) if file.endswith('.TIF')]
    all_bands = [rasterio.open(os.path.join(data_path, file)).read(1) for file in all_files]
    hyperspectral_cube = np.dstack(all_bands)
    return hyperspectral_cube

def segment_hyperion_image(hyperspectral_cube):
    data_norm = (hyperspectral_cube - hyperspectral_cube.min()) / (hyperspectral_cube.max() - hyperspectral_cube.min())
    data_2d = data_norm.reshape((hyperspectral_cube.shape[0], -1)).T
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(data_2d)
    segmented_image = labels.reshape(hyperspectral_cube.shape[:-1])
    return segmented_image

def visualize_segmented_image(segmented_image):
    plt.imshow(segmented_image, cmap='tab20b')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

def extract_spectral_profiles(hyperspectral_cube, segmented_image):
    wavelengths = np.linspace(0.357, 2.576, hyperspectral_cube.shape[2])
    spectral_profiles = []

    num_clusters = len(np.unique(segmented_image))
    for cluster_id in range(num_clusters):
        mask = (segmented_image == cluster_id)
        cluster_pixels = hyperspectral_cube[mask]
        avg_profile = cluster_pixels.mean(axis=0)
        spectral_profiles.append(avg_profile)
    
    return wavelengths, spectral_profiles

def visualize_spectral_profiles(wavelengths, spectral_profiles):
    plt.figure(figsize=(12, 6))
    for cluster_id, profile in enumerate(spectral_profiles):
        plt.plot(wavelengths, profile, label=f'Cluster {cluster_id}', marker='o')
    plt.title('Average Spectral Profiles for Each Cluster')
    plt.xlabel('Wavelength (micrometers)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_spectral_profiles(wavelengths, average_reflectance_values, output_file='average_spectral_profile.csv'):
    output_df = pd.DataFrame({
        'Wavelength (micrometers)': wavelengths,
        'Average Reflectance': average_reflectance_values
    })
    output_df.to_csv(output_file, index=False)

def visualize_hyperion_eo1(data_path=None):
    if not data_path:
        data_path = input("Please enter the path to the directory containing Hyperion data: ")

    hyperspectral_cube = load_hyperion_data(data_path)
    segmented_image = segment_hyperion_image(hyperspectral_cube)
    visualize_segmented_image(segmented_image)
    wavelengths, spectral_profiles = extract_spectral_profiles(hyperspectral_cube, segmented_image)
    visualize_spectral_profiles(wavelengths, spectral_profiles)
    average_reflectance_values = [np.mean(band) for band in spectral_profiles]
    save_spectral_profiles(wavelengths, average_reflectance_values)

# ... [The hyperion functions above]

# Phenology Visualization Function
def visualize_phenology(directory=None):
    if not directory:
        directory = input("Please enter the path to the directory containing Phenology data: ")
    
    for filename in os.listdir(directory):
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            f = os.path.join(directory, filename)
            try:
                with h5py.File(f, 'r') as file:
                    plt.figure(figsize=(6, 4))
                    plt.imshow(file['HDFEOS']['GRIDS']['Cycle 1']['Data Fields']['PGQ_Growing_Season_1'][:], cmap='jet')
                    plt.title('PGQ_Growing_Season')
                    plt.show()

                    plt.figure(figsize=(6, 4))
                    plt.imshow(file['HDFEOS']['GRIDS']['Cycle 1']['Data Fields']['GLSP_QC_1'][:], cmap='jet')
                    plt.title('GLSP_QC')
                    plt.show()

                    plt.figure(figsize=(6, 4))
                    plt.imshow(file['HDFEOS']['GRIDS']['Cycle 1']['Data Fields']['Greenness_Agreement_Growing_Season_1'][:], cmap='jet')
                    plt.title('Greenness_Agreement_Growing_Season')
                    plt.show()
            except Exception as e:
                print(f"Error visualizing phenology data from {filename}: {e}")

# NDVI Visualization Function
def visualize_ndvi(directory=None):
    if not directory:
        directory = input("Please enter the path to the directory containing NDVI data: ")
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        try:
            if os.path.isfile(f):
                with h5py.File(f, 'r') as file:
                    data = file['HDFEOS']['GRIDS']['VIIRS_Grid_16Day_VI_1km']['Data Fields']['1 km 16 days NDVI'][:]
                    year = int(filename[10:14])
                    day_of_year = float(filename[14:16])
                    date = datetime(year, 1, 1) + timedelta(int(day_of_year) - 1)
                    plt.figure()
                    plt.imshow(data)
                    plt.colorbar(label='NDVI')
                    plt.title(f"Date of acquisition: {date}")
                    plt.grid(False)
                    plt.axis('off')
                    plt.show()
        except Exception as e:
            print(f"Error visualizing NDVI data from {filename}: {e}")

# MODIS Visualization Function
def visualize_modis(directory=None):
    if not directory:
        directory = input("Please enter the path to the directory containing MODIS data: ")
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        try:
            ds = Dataset(f, 'r')
            data = ds.variables['water_mask'][:]
            plt.figure(figsize=(10, 6))
            plt.imshow(data, cmap='gray')
            plt.show()
        except Exception as e:
            print(f"Error visualizing MODIS data from {filename}: {e}")

# Photosynthesis Visualization Function
def visualize_photosynthesis(directory=None):
    if not directory:
        directory = input("Please enter the path to the directory containing Photosynthesis data: ")
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        try:
            ds = xr.open_dataset(f, engine='netcdf4')
            data = ds['GMT_1200_PAR']
            plt.figure(figsize=(6, 4))
            plt.imshow(data, cmap='jet')
            plt.xlabel('XDim:MODISRAD')
            plt.ylabel('YDim:MODISRAD')
            plt.title('Total PAR at 12:00')
            plt.show()
        except Exception as e:
            print(f"Error visualizing photosynthesis data from {filename}: {e}")

