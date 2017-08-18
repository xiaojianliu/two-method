# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 09:21:18 2017

@author: xiaojian
"""

import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
import matplotlib.pyplot as plt
from dateutil.parser import parse
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
from matplotlib.dates import date2num,num2date
def haversine(lon1, lat1, lon2, lat2): 
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """   
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    #print 34
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * atan(sqrt(a)/sqrt(1-a))   
    r = 6371 
    d=c * r
    #print type(d)
    return d
lon=np.load('lonnnnn201360x.npy')
lat=np.load('lattttt201360x.npy')
time=np.load('time201360x.npy')
FNCL='necscoast_worldvec.dat'
CL=np.genfromtxt(FNCL,names=['lon','lat'])
fig,axes=plt.subplots(1,2,figsize=(14,5))#,sharex=True)

aa=0
a7=0
a8=0
for a in np.arange(len(lon)):
    
    if len(lon[a])==1:
        continue
    else:
        #aa=aa+1
        if time[a][1]>datetime(2013,12,1) and time[a][-1]<datetime(2013,11,17):
            a7=a
            axes[0].scatter(lon[a][0],lat[a][0],color='red')
            axes[0].scatter(lon[a][-1],lat[a][-1],color='green')
            axes[0].plot(lon[a],lat[a],color='yellow',linewidth=0.5)
        if time[a][1]<datetime(2013,12,1) and time[a][-1]<datetime(2013,11,17):
            axes[0].scatter(lon[a][0],lat[a][0],color='red')
            axes[0].scatter(lon[a][-1],lat[a][-1],color='green')
            axes[0].plot(lon[a],lat[a],color='yellow',linewidth=0.5)
axes[0].scatter(lon[a7][0],lat[a7][0],color='red',label='end point')
axes[0].scatter(lon[a7][-1],lat[a7][-1],color='green',label='start point')
axes[0].plot(lon[a7],lat[a7],color='yellow',linewidth=0.5,label='track')
axes[0].legend()
axes[0]. plot(CL['lon'],CL['lat'],linewidth=0.7)
axes[0].axis([-71.0,-69.8,41.63,43.5])
for a in np.arange(len(lon)):
    
    if len(lon[a])==1:
        continue
    else:
        
        if time[a][1]>datetime(2013,12,1) and time[a][-1]<datetime(2013,11,17):
            aa=aa+1
            axes[1].scatter(lon[a][0],lat[a][0],color='red')
            axes[1].scatter(lon[a][-1],lat[a][-1],color='green')
        if time[a][1]<datetime(2013,12,1) and time[a][-1]<datetime(2013,11,17):
            aa=aa+1
            plt.scatter(lon[a][0],lat[a][0],color='red')
            plt.scatter(lon[a][-1],lat[a][-1],color='cyan')
axes[1]. plot(CL['lon'],CL['lat'],linewidth=0.7)
axes[1].axis([-71.0,-69.8,41.63,43.5])
axes[0].xaxis.tick_top() 
axes[1].xaxis.tick_top() 
lon1=np.load('lonnnnn201260.npy')
lat1=np.load('lattttt201260.npy')
time1=np.load('timedian201260.npy')
for a in np.arange(len(lon1)):
    
    if len(lon1[a])==1:
        continue
    else:
        #aa=aa+1
        if time1[a][1]>datetime(2012,12,1) and time1[a][-1]<datetime(2012,11,17):
            axes[0].scatter(lon1[a][0],lat1[a][0],color='red')
            axes[0].scatter(lon1[a][-1],lat1[a][-1],color='green')
            axes[0].plot(lon1[a],lat1[a],color='yellow',linewidth=0.5)
        if time1[a][1]<datetime(2012,12,1) and time1[a][-1]<datetime(2012,11,17):
            axes[0].scatter(lon1[a][0],lat1[a][0],color='red')
            axes[0].scatter(lon1[a][-1],lat1[a][-1],color='green')
            axes[0].plot(lon1[a],lat1[a],color='yellow',linewidth=0.5)
a4=0
a5=0
for a1 in np.arange(len(lon1)):
    l=0
    if len(lon1[a1])==1:
        continue
    else:
        
        if time1[a1][1]>datetime(2012,12,1) and time1[a1][-1]<datetime(2012,11,17):
            aa=aa+1
            a4=a1
            axes[1].scatter(lon1[a1][0],lat1[a1][0],color='red')
            axes[1].scatter(lon1[a1][-1],lat1[a1][-1],color='green')
            for a2 in np.arange(len(lon1[a1])-1):
                l=l+haversine(lon1[a1][a2],lat1[a1][a2],lon1[a1][a2+1],lat1[a1][a2+1])
            #c=plt.Circle((lon1[a1][-1],lat1[a1][-1]),0.18*l*0.009009,color='green',alpha=0.2)
            #axes[1].add_patch(c)
        if time1[a1][1]<datetime(2012,12,1) and time1[a1][-1]<datetime(2012,11,17):
            aa=aa+1
            a5=a1
            plt.scatter(lon1[a1][0],lat1[a1][0],color='red')
            plt.scatter(lon1[a1][-1],lat1[a1][-1],color='cyan')
            for a2 in np.arange(len(lon1[a1])-1):
                l=l+haversine(lon1[a1][a2],lat1[a1][a2],lon1[a1][a2+1],lat1[a1][a2+1])
            #c=plt.Circle((lon1[a1][-1],lat1[a1][-1]),0.18*l*0.009009,color='yellow',alpha=0.2)
            #axes[1].add_patch(c)
axes[1].scatter(lon1[a1][0],lat1[a1][0],color='red',label='end point')
axes[1].scatter(lon1[a5][-1],lat1[a5][-1],color='cyan',label='start point')#(November)')
axes[1].scatter(lon1[a4][-1],lat1[a4][-1],color='green',label='start point')#(December)')
axes[1].set_xlabel('b')
axes[0].set_xlabel('a')
axes[1].set_yticklabels([])

axes[1].legend(loc='best')
plt.savefig('track2012-2013',dpi=400)
