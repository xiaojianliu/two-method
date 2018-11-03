# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:19:50 2018

@author: xiaojian
"""
from math import radians, cos, sin, atan, sqrt 
import numpy as np
from matplotlib.path import Path
from scipy import  interpolate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
lon=-70.087
lat=41.767
size=0.03
r=0.010
track_days=7
km=1.5
dgree=km/111.0
endtime =datetime(2014,11,22,12,0,0)
starttime=endtime-timedelta(hours=track_days*24)

FNCL='necscoast_worldvec.dat'
CL=np.genfromtxt(FNCL,names=['lon','lat'])
cl=dict(lon=[],lat=[])
for a in np.arange(len(CL['lon'])):
    if CL['lon'][a]>lon-size and CL['lon'][a]<lon+size and CL['lat'][a]<lat+size and CL['lat'][a]>lat-size:
        cl['lon'].append(CL['lon'][a])
        cl['lat'].append(CL['lat'][a])

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(cl['lon'],cl['lat'],color='black')
ax.scatter(lon,lat,color='red')
ax.axis([lon-size,lon+size,lat-size,lat+size])
dis=[]
for a in np.arange(len(cl['lon'])):
    d=haversine(cl['lon'][a],cl['lat'][a],lon,lat)
    dis.append(d)
index=np.argmin(dis)
plt.scatter(cl['lon'][index],cl['lat'][index],color='green')
lonz=cl['lon'][index]
latz=cl['lat'][index]
k=(latz-lat)/(lonz-lon)

lon_w1=lonz+np.sqrt(dgree**2/(1+k**2))
lon_w2=lonz-np.sqrt(dgree**2/(1+k**2))
d1=abs(lon_w1-lon)
d2=abs(lon_w2-lon)
if d1>d2:
    lon_w=lon_w1
    lat_w=k*(lon_w-lonz)+latz
else:
    lon_w=lon_w2
    lat_w=k*(lon_w-lonz)+latz
ax.scatter(lon_w,lat_w,color='yellow')
ax.plot([lon,lon_w],[lat,lat_w])
plt.savefig('method1')
plt.show()

