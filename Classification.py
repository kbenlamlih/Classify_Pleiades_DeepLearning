import multiprocessing
from multiprocessing import shared_memory
from multiprocessing import Pool
from multiprocessing import Array
from multiprocessing import Lock
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdalconst
from datetime import datetime 

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal_array
import matplotlib.patches as mpatches
from numpy import asarray
import tensorflow as tf
from keras.models import model_from_json
from datetime import datetime 
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes
from concurrent.futures import ProcessPoolExecutor
import ctypes  
import os
import sys
import time


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
        
        
patch_width = 25               
patch_height = patch_width
pad= 12

json_file = open('Modeles/m3_niveau2_best.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
cnn.load_weights("Modeles/m3_niveau2_best.h5")

epsg = 2975
RasterFormat ='GTiff'
src= 'Phase2/2018_test.tif'
Raster =gdal.Open(src, gdal.GA_ReadOnly)
Projection = Raster.GetProjectionRef()
rasterArray = gdal_array.LoadFile(src)
rasterArray = rasterArray.transpose(1, 2, 0)
X_test = asarray(rasterArray, dtype='uint8')
img_height, img_width, band = X_test.shape

def shared_array():
    """
    Form a shared memory numpy array.
    
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing 
    """
    
    shared_array_base = multiprocessing.Array(ctypes.c_ubyte, img_height*img_width)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(img_height,img_width)
    return shared_array

X_class = np.pad(X_test, ((pad,pad), (pad,pad), (0, 0)), 'reflect')
        
shm = shared_memory.SharedMemory(create=True, size=X_class.nbytes)
X_class_shared = np.ndarray(X_class.shape, dtype=X_class.dtype, buffer=shm.buf)
X_class_shared =X_class.copy()


shm2 = shared_memory.SharedMemory(create=True, size=X_test.nbytes)
classif = np.ndarray(X_test.shape, dtype=X_test.dtype, buffer=shm2.buf)

def intervalles(NUM_PROCESSES):
    N = X_test.shape[0]
    P = (NUM_PROCESSES + 1) 
    partitions = list(zip(np.linspace(0, N, P, dtype=int)[:-1], np.linspace(0, N, P, dtype=int)[1:]))
    work = partitions[:-1]
    work.append((partitions[-1][0], partitions[-1][1]))


    M = X_test.shape[1]
    partitions2 = list(zip(np.linspace(0, M, P, dtype=int)[:-1],
    np.linspace(0, M, P, dtype=int)[1:]))

        # Final range of indices should end +1 past last index for completeness
    work2 = partitions2[:-1]
    work2.append((partitions2[-1][0], partitions2[-1][1] ))


    work3=list()
    for i in work:
        for j in work2:
            n= list((j,i))
            work3.append(n)
                
    return work3    
def array2raster(newRasterfn, dataset, array, dtype):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    #band_num=1
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols,rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
def predict(i, j):
    tf.keras.backend.clear_session()
    x=j 
    x_bis = j+patch_height
    y=i
    y_bis=i +patch_width 
    patch = X_class_shared[x:x_bis,y:y_bis, :]
    X = np.expand_dims(patch, axis=0)
    test = cnn.predict(X)
    rounded = np.argmax(test, axis=1)
            
    return rounded 
 


def dispatch_jobs(data, job_number):
    #jobs = []
    results= dict()
    pool =Pool(job_number)
    for i in range(img_width):
        for j in range(img_height):
            results[(i,j)]=pool.apply_async(predict, args=(i,j))
    pool.close()
    pool.join()
    
    for i, j in results.keys(): 
        classif[j,i]=results[(i,j)].get()
        
    array2raster('Test_niveau2/227x227_pour_temps.tif',Raster,classif, 'Byte')




if __name__ == "__main__":
    #NUM_PROCESSES=20
    NUM_PROCESSES = input("Nombre de processus:\n")  
    NUM_PROCESSES = int(NUM_PROCESSES)
    work3 = intervalles(NUM_PROCESSES)
    start_time = time.time()
    dispatch_jobs(work3,NUM_PROCESSES)
    print("--- %s seconds ---" % (time.time() - start_time))

    