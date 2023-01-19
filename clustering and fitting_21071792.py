# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn import preprocessing
import itertools as iter


def read_and_filter_csv(file_name):
    """
    This function is used to read the csv file from the directory and to import
    the data for the Density clustering.

    file_name :- the name of the csv file with data.   
    """

    file_data = pd.read_csv(file_name)
    dataFr = pd.DataFrame(file_data)
    print(dataFr)
    dataFr = dataFr[['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']]
    dataFr = dataFr.iloc[61:76]
    return file_data, dataFr




#pair plot comparing the two data
def pair_plot():
    
    data_frame_plot = pd.DataFrame()
    greengas = []
    methane  = []
    
    for i in methaneGas_data :
        methane.extend(methaneGas_data[i])

    for i in greenHouse_data:
        greengas.extend(greenHouse_data[i])

    data_frame_plot['greengas'] = greengas
    data_frame_plot['methane'] = methane

    type(data_frame_plot)

    sns.pairplot(data_frame_plot[['methane','greengas']])

    '''function for finding K means clusttering'''
    kmeans1 = KMeans(n_clusters=3, random_state=0).fit(data_frame_plot[['methane','greengas']])

    kmeans1.inertia_
    kmeans1.cluster_centers_

    data_frame_plot['cluster'] = kmeans1.labels_
    
    
    return data_frame_plot



def scatter_plot(data_k_plot):
    
    '''plot for K means clusttering before normalisation'''
    plt.figure()
    sns.scatterplot(x = 'greengas', y = 'methane' , hue='cluster', data = data_k_plot)
    plt.title("K-Means before normalisation")
    plt.show()

    data_k = data_fr.drop(['cluster'], axis = 1)
    '''function called for clusttering'''
    names = ['greengas','methane']
    a = preprocessing.normalize(data_k, axis=0)
    data_aft_k = pd.DataFrame(a,columns=names)
    kmeans2 = KMeans(n_clusters=3, random_state=0).fit(data_aft_k[['methane','greengas']])
    kmeans2.inertia_
    kmeans2.cluster_centers_
    data_aft_k['cluster'] = kmeans2.labels_
    '''cluster shown along the data'''
    '''plot for K means clusttering after normalisation'''
    plt.figure()
    sns.scatterplot(x = 'greengas', y = 'methane' , hue='cluster', data = data_aft_k)
    plt.title("K-Means after normalisation")
    plt.show()
    return

'''function to calculate the error limits'''

def func(x,a,b,c):
    return a * np.exp(-(x-b)**2 / c)


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
   
    """
 
   
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
   
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
       
    pmix = list(iter.product(*uplow))
   
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
       
    return lower, upper



def scatter_plot():
    
    data_n = pd.DataFrame()
    greengas = []
    methane  = []
    
    for i in methaneGas_data :
        methane.extend(methaneGas_data[i])

    for i in greenHouse_data:
        greengas.extend(greenHouse_data[i])

    data_frame_plot['greengas'] = greengas
    data_frame_plot['methane'] = methane
    
    '''plot for scattering'''
    plt.scatter(data_ele['consumption'],data_ele['access'])
    plt.title('Scatter plot before curve fitting')
    plt.ylabel('Consumtion')
    plt.xlabel('Access to electricity')
    plt.show()
    return

'''adding an exponential function'''
def expoFunc(x,a,b):
    return a**(x+b)

x_data = data_ele['consumption']
y_data = data_ele['access']
popt, pcov = curve_fit(expoFunc,x_data,y_data,p0=[1,0])
popt
a_opt, b_opt = popt
x_mod = np.linspace(min(x_data),max(x_data),100)
y_mod = expoFunc(x_mod,a_opt,b_opt)

'''plot for scattering after fitting the curve'''
plt.scatter(x_data,y_data)
plt.plot(x_mod,y_mod,color = 'r')
plt.title('Scatter plot after curve fitting')
plt.ylabel('Consumtion')
plt.xlabel('Access to electricity')
plt.savefig("curvefit_after.png")
plt.show()




countries = ['Austria', 'Belgium', 'Denmark', 'France', 'Ireland', 'Portugal']

#dataset with data regarding the total greenhouse emissions from a period of year 1994-2012
org_green_data, greenHouse_data = read_and_filter_csv("Total greenhouse gas emissions.csv")
    

#dataset with data regarding the total methane gas emission from a period of year 1994-2012
org_methane_data, methaneGas_data = read_and_filter_csv("Methane emissions.csv")

data_fr = pair_plot()
scatter_plot(data_fr)
