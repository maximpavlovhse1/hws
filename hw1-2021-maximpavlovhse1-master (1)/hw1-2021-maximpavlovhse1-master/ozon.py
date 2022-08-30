import argparse
import json
from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt




parser = argparse.ArgumentParser()







parser.add_argument('latitude',  metavar='LAT', type = float, help='Latitude, deg')
parser.add_argument('longitude', metavar='LON', type = float, help='Longitude, deg')


if __name__ == "__main__":
    args = parser.parse_args()
    
        
            





with netcdf.netcdf_file("MSR-2.nc", mmap = False) as netcdf_file:
    print(netcdf_file.dimensions)
    for v in netcdf_file.variables:
        var = netcdf_file.variables[v]
        print(v, var.dimensions, var.data.shape)


    lon_index = np.searchsorted(netcdf_file.variables['longitude'].data, args.longitude)
    lat_index = np.searchsorted(netcdf_file.variables['latitude'].data, args.latitude)
    Max_all = np.around(netcdf_file.variables['Average_O3_column'].data[:,lat_index, lon_index].max(), 1)
    Min_all = np.around(netcdf_file.variables['Average_O3_column'].data[:,lat_index, lon_index].min(), 1)
    Mean_all = np.around(netcdf_file.variables['Average_O3_column'].data[:,lat_index, lon_index].mean(), 1)
    
    
    
    Jan_max =np.around(netcdf_file.variables['Average_O3_column'].data[::12,lat_index, lon_index].max(), 1)
    Jan_min = np.around(netcdf_file.variables['Average_O3_column'].data[::12,lat_index, lon_index].min(), 1)
    Jan_mean = np.around(netcdf_file.variables['Average_O3_column'].data[::12,lat_index, lon_index].mean(), 1)
    
       
    
    Jul_max = np.around(netcdf_file.variables['Average_O3_column'].data[6::12,lat_index, lon_index].max(), 1)
    Jul_min = np.around(netcdf_file.variables['Average_O3_column'].data[6::12,lat_index, lon_index].min(), 1)
    Jul_mean = np.around(netcdf_file.variables['Average_O3_column'].data[6::12, lat_index, lon_index].mean(), 1)
  

   
    
    x = netcdf_file.variables['time'].data
    y = netcdf_file.variables['Average_O3_column'].data[:,lat_index, lon_index]
    
    x1 = netcdf_file.variables['time'].data[::12]
    y1 = netcdf_file.variables['Average_O3_column'].data[::12,lat_index, lon_index]
    
    x2 = netcdf_file.variables['time'].data[6::12]
    y2 = netcdf_file.variables['Average_O3_column'].data[6::12, lat_index, lon_index]
    




plt.plot(x2, y2, color = "black")
plt.plot(x, y, color = "green")
plt.plot(x1, y1, color = "red")
plt.xlabel('Время по месяцам')
plt.ylabel('Содержание озона')
plt.legend(['за июли','за все время', 'за январи'])
plt.savefig("ozon.png")

a = [float(args.latitude), float(args.longitude)]
d = {"coordinates": a,
     "jan": {"min": float(Jan_min), 
             "max": float(Jan_max), 
             "mean": float(Jan_mean)}, 
     "jul":{"min": float(Jul_min), 
            "max": float(Jul_max), 
            "mean": float(Jul_mean)}, 
     "all": { "min": float(Min_all),
            "max": float(Max_all), 
            "mean": float(Mean_all)}}


with open("ozon.json", "w") as f: 
    json.dump(d, f, indent = 2)


