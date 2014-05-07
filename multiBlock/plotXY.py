
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np
import time

grid = np.load('XY.npy')

#fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

#ax1.imshow(grid, extent=[0,100,0,1])
#ax1.set_title('Default')
#Z=np.array(((1,2,3,4,5),(4,5,6,7,8),(7,8,9,10,11)))
#im = plt.imshow(Z, cmap='hot')
#plt.colorbar(im, orientation='horizontal')
plt.ion()
plt.imshow(grid[0,:,:], extent=[0,1,0,1], aspect='auto', vmin=0, vmax=3)
plt.set_cmap('gray')
plt.colorbar()
plt.show()

#ax2.imshow(grid, extent=[0,100,0,1], aspect='auto')
#plt.draw()
for j in range(0,grid.shape[0]):
    #im.set_data(grid[j,:,:])
    plt.imshow(grid[j,:,:], extent=[0,1,0,1], aspect='auto', vmin=0, vmax=3)
    plt.set_cmap('gray')
#    plt.set_data(grid[j,:,:])
    plt.draw()
#    time.sleep(0.1)
#ax2.set_title('Auto-scaled Aspect')

#ax3.imshow(grid, extent=[0,100,0,1], aspect=100)
#ax3.set_title('Manually Set Aspect')

