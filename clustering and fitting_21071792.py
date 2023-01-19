# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn import preprocessing
import itertools as iter


def read_and_filter_csv(file_name):
    """
    This function is used to read the csv file from the directory and to import
    the data for the For curve and fitting

    file_name :- the name of the csv file with data.   
    """

    file_data = pd.read_csv(file_name)
    dataFr = pd.DataFrame(file_data)
    dataFr = dataFr[['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
                     '2008', '2009', '2010', '2011']]
    dataFr = dataFr.iloc[61:76]
    print(file_data)
    print(dataFr)
    return file_data, dataFr


# pair plot comparing the two data
def pair_plot():
    """
    The function helps is creating the pair plot of greenhouse gas emission
    and the methane gas emission.As well to find the k- means clusters of the data
    """

    data_frame_plot = pd.DataFrame()
    greengas = []
    methane = []

    for i in methaneGas_data:
        methane.extend(methaneGas_data[i])

    for i in greenHouse_data:
        greengas.extend(greenHouse_data[i])

    data_frame_plot['greengas'] = greengas
    data_frame_plot['methane'] = methane

    type(data_frame_plot)

    # Plottting the data
    sns.pairplot(data_frame_plot[['methane', 'greengas']])
    sns.set(font_scale = 1.5)
    plt.savefig("graphs//pair plot.png", dpi =215)

    # function for finding K means clusttering
    kmeans1 = KMeans(n_clusters=3, random_state=0).fit(
        data_frame_plot[['methane', 'greengas']])
    kmeans1.inertia_
    kmeans1.cluster_centers_
    data_frame_plot['cluster'] = kmeans1.labels_
    print(data_frame_plot)
    return data_frame_plot


# Scatter plot with K means clustering before and afer normalization
def scatter_k_plot(data_k_plot):
    """ 
    The function is used to find the K-mean clusters and also to construct the 
    scatter plot with and without normalizing the data.
    data_k_plot :- the data frame with greenhouse and methane emission data
    """

    # plot for K means clusttering before normalisation
    plt.figure()
    sns.scatterplot(x='greengas', y='methane', hue='cluster', data=data_k_plot)
    plt.title("K-Means before normalisation", fontsize = 20, color='purple')
    plt.savefig("graphs//Scatter of K-mean.png", dpi =215)
    plt.show()

    data_k = data_fr.drop(['cluster'], axis=1)
    names = ['greengas', 'methane']
    a = preprocessing.normalize(data_k, axis=0)
    data_aft_k = pd.DataFrame(a, columns=names)
    kmeans2 = KMeans(n_clusters=3, random_state=0).fit(
        data_aft_k[['methane', 'greengas']])
    kmeans2.inertia_
    kmeans2.cluster_centers_
    data_aft_k['cluster'] = kmeans2.labels_

    # plot for K means clusttering after normalisation
    plt.figure()
    sns.scatterplot(x='greengas', y='methane', hue='cluster', data=data_aft_k)
    plt.title("K-Means after normalisation", fontsize = 20, color='purple')
    plt.savefig("graphs//Scatter of K-mean normalized.png", dpi =215)
    plt.show()
    return


# function to calculate the error limits'''
def func(x, a, b, c):
    return a * np.exp(-(x-b)**2 / c)


# finding upper and lowe limit or ranges of the function
def err_ranges(x, func, param, sigma):
    """
    The function helps to calculate the upper and lower limits for the function,
    parameters and sigmas for single value or array x. Function values are calculated 
    for all the combinations of positive/negative sigma and the minimum and maximum 
    is determined whcich can be used for all number of parameters and sigmas >=1.  
    """

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper


# Scatter plot of data without normalizing
def scatter_plot():
    """
    The function is used to create scatter plot without fitting the curve
    """

    data_n = pd.DataFrame()
    greengas = []
    methane = []

    for i in methaneGas_data:
        methane.extend(methaneGas_data[i])

    for i in greenHouse_data:
        greengas.extend(greenHouse_data[i])

    data_n['greengas'] = greengas
    data_n['methane'] = methane

    # plot for scattering
    plt.scatter(data_n['greengas'], data_n['methane'])
    plt.title('Scatter plot of emissions without curve fitting', fontsize=20, color='purple')
    plt.ylabel('Green gas Emission--->', fontsize=16)
    plt.xlabel('Methane gas Emission--->', fontsize=16)
    plt.savefig("graphs//scatter plot.png", dpi =215)
    plt.show()
    return

# adding an exponential function'''
def expoFunc(x, a, b):
    return a**(x+b)


# Fitting the curve without scatter
def curve_fitting():
    """ 
    The function is used to fit a curve with the data without clustering the data
    """

    xaxis_data = data_fr['greengas']
    yaxis_data = data_fr['methane']
    popt, pcov = curve_fit(expoFunc, xaxis_data, yaxis_data, p0=[1, 0])
    ab_opt, bc_opt = popt
    x_mod = np.linspace(min(xaxis_data), max(xaxis_data), 100)
    y_mod = expoFunc(x_mod, ab_opt, bc_opt)

    # plot for scattering after fitting the curve
    plt.plot(x_mod, y_mod, color='r')
    plt.title('Fitting the curve without cluster points', fontsize=20, color='purple')
    plt.ylabel('green gas emission--->', fontsize=16)
    plt.xlabel('Methane gas emission--->',fontsize=16)
    plt.savefig("graphs//curve_Fit.png", dpi =215)
    plt.show()
    return


# The scatter plot with both cluster and curve fit
def scatter_plot_n():
    """ 
    The function is used to fit the curve over the cluster for the same data.
    """

    xaxis_data = data_fr['greengas']
    yaxis_data = data_fr['methane']
    popt, pcov = curve_fit(expoFunc, xaxis_data, yaxis_data, p0=[1, 0])
    ab_opt, bc_opt = popt
    x_mod = np.linspace(min(xaxis_data), max(xaxis_data), 100)
    y_mod = expoFunc(x_mod, ab_opt, bc_opt)

    # plot for scattering after fitting the curve
    plt.scatter(xaxis_data, yaxis_data)
    plt.plot(x_mod, y_mod, color='r')
    plt.title('Scatter plot with the curve fitting', fontsize=20, color='purple')
    plt.ylabel('green gas emission--->', fontsize=16)
    plt.xlabel('Methane gas emission--->', fontsize=16)
    plt.savefig("graphs//curve_and_Cluster.png", dpi =215)
    plt.show()
    return

#Sorting the data for required countries and years
countries = ['Austria', 'Belgium', 'Denmark', 'France', 'Ireland', 'Portugal']

# data for total greenhouse emissions from a period of year 2000-2011
org_green_data, greenHouse_data = read_and_filter_csv(
    "Total greenhouse gas emissions.csv")


# data for total methane gas emission from a period of year 2000-2011
org_methane_data, methaneGas_data = read_and_filter_csv(
    "Methane emissions.csv")

# Invoking the functions
data_fr = pair_plot()
scatter_k_plot(data_fr)
scatter_plot()
curve_fitting()
scatter_plot_n()
