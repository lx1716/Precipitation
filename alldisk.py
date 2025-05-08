import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap


data_fulldisk = np.loadtxt("pre.txt").reshape(203,203, 32, 32)


def calWeight(d,k):
    '''
    :param d: Diameter of overlapping area
    :param k: Weight calculation parameter
    :return:
    '''
    x = np.arange(-d/2,d/2)
    y = 1/(1+np.exp(-k*x/d))
    return y

def imgFusion(img1,img2,overlap,left_right=True):
 

    w = calWeight(overlap,5)    

    if left_right: 
        row,col = img1.shape
        row1,col1 = img2.shape
        img_new = np.zeros((row,col+col1-overlap))
        img_new[:,:col] = img1
        w_expand = np.tile(w,(row,1))  
        img_new[:,col-overlap:col] = (1-w_expand)*img1[:,col-overlap:col]+w_expand*img2[:,:overlap]
        img_new[:,col:]=img2[:,overlap:]
    else:   
        row,col = img1.shape
        row1,col1 = img2.shape
        img_new = np.zeros((row+row1-overlap,col))
        img_new[:row,:] = img1
        w = np.reshape(w,(overlap,1))
        w_expand = np.tile(w,(1,col))
        img_new[row-overlap:row,:] = (1-w_expand)*img1[row-overlap:row,:]+w_expand*img2[:overlap,:]
        img_new[row:,:] = img2[overlap:,:]
    return img_new

def row_merge(data_fulldisk,row,overlap,numberh1,p):
    for col in range(numberh1-1):
        if col==0:
            img1 = data_fulldisk[p,row,col,:,:]
        else:
            img1 = img_new
        img2 = data_fulldisk[p,row,col+1,:,:]
        img_new = imgFusion(img1,img2,overlap,left_right=True)
    return img_new

def row_merge2(data_fulldisk,row,overlap,numberh1):
    for col in range(numberh1-1):
        if col==0:
            img1 = data_fulldisk[row,col,:,:]
        else:
            img1 = img_new
        img2 = data_fulldisk[row,col+1,:,:]
        img_new = imgFusion(img1,img2,overlap,left_right=True)
    return img_new

overlap_size=16
model_size=32
boxs=203
for i in range(boxs-1):
        if i==0:
            img_col1=row_merge2(data_fulldisk,0,overlap_size,boxs)
        else:
            img_col1=img_new_col
        img_col2=row_merge2(data_fulldisk,i+1,overlap_size,boxs)
        img_new_col = imgFusion(img_col1,img_col2,overlap_size,left_right=False)

img_new_col[img_new_col>0.151]=0.151
img_new_col[img_new_col<0.001]=0.0
img_new_col=img_new_col*100.0

img_new_col_flipped = img_new_col[::-1, :]

max_value = 10


img_new_col = img_new_col.reshape(3264, 3264)
start_lat = -81.5
start_lon = 51.5
delta = 0.05 

lon = np.arange(start_lon, start_lon + 3264 * delta, delta)
lat = np.arange(start_lat, start_lat + 3264 * delta, delta) 
Lon, Lat = np.meshgrid(lon, lat)


latcorners = [start_lat, start_lat + 3264 * delta]
loncorners = [start_lon, start_lon + 3264 * delta]
fig = plt.figure(figsize=(12, 10))
m = Basemap(projection='ortho',
            lat_0=0, lon_0=123.5)
m.drawlsmask(land_color="#FFFFCC",
             ocean_color="#ADD8E6",
             resolution='l')

m.drawcoastlines(linewidth=0.5, linestyle='solid', color='k', antialiased=True)
m.drawstates(linewidth=0.5, linestyle='solid', color='k', antialiased=True)


parallels = np.arange(latcorners[0], latcorners[1]+0.1, 20)
m.drawparallels(parallels, labels=[True, False, True, False], linewidth=0.5, fontsize=14, fontname="Times New Roman", weight="bold")
meridians = np.arange(loncorners[0], loncorners[1]+0.1, 20)
m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.5, fontsize=14, fontname="Times New Roman", weight="bold")

cmap = plt.cm.get_cmap("rainbow") 


boundaries = np.array([0.1,0.5, 1, 1.5,2,2.5,3,3.5,4,4.5, 5, 6,7,8,9, 10]) 
clevs = boundaries 

im = m.contourf(Lon, Lat, img_new_col_flipped, clevs, cmap=cmap, latlon=True, extend='max')

#cbar = fig.colorbar(im, ax=m.ax, orientation='vertical', pad=0.05)
cbar = fig.colorbar(im, ax=m.ax, orientation='vertical', pad=0.05, ticks=boundaries)
cbar.set_ticklabels(boundaries.astype(str)) 
labels = cbar.ax.yaxis.get_ticklabels()

cbar.ax.tick_params(labelsize = '12') 
cbar.set_label('mm/h',fontsize =12)


plt.tick_params(top=False, bottom=True, left=True, right=False)
plt.tick_params(labeltop=False, labelleft=False, labelright=True, labelbottom=True)


plt.title("pre", fontsize=26, fontweight="bold", loc="center")


plt.show()
