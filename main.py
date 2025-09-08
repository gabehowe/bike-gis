import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
from scipy import ndimage
import time
import types
import cv2
from PIL import Image


def reconstruct_path(cameFrom: dict, current):
    total_path = [current]
    while current in cameFrom:
        current = cameFrom[current]
        total_path.append(current)
    return total_path[::-1]

# return slope between points

def d(current, point, weight):
    if not (point[0] in range(0,grad.shape[0]) and point[1] in range(0, grad.shape[1])):
        return 1000000000
    if grad[point][0] == 10000:
        print('in building')
        return 1000000000000000
    vector = (point[0] - current[0], point[1] - current[1])
    rv =  grad[point] @ vector * weight
    return max(rv + 50000, 0)

def h(it, goal):
    return np.hypot(*(goal[0] - it[0], goal[1] - it[1]))
ACCEPTABLE_TIME = 15
STEP = 8
def a_star(start: tuple, goal: tuple, weight):
    #h = lambda it: np.abs(goal[0] - it[0]) + np.abs( goal[1] - it[1])
    # ndarrays are tuples
    openSet = set([start])
    closedSet = set()
    cameFrom = {}
    gScore = {} # default value of infinity
    gScore[start] = 0
    min_hv = np.inf

    fScore = {} # default value of infinity
    fScore[start] = h(start, goal) * 0.01
    hv = np.inf
    directions = [(x*STEP,y*STEP) for x in range(-1,2) for y in range(-1,2)]
    init_time = time.perf_counter_ns()
    timer = init_time
    while len(openSet) > 0:

        if min_hv <= 30:
            return reconstruct_path(cameFrom, current)
        current = min(openSet, key=lambda x: fScore[x])
        ct = time.perf_counter_ns()
        if (ct-init_time) / 1_000_000_000 > ACCEPTABLE_TIME:
            weight *= 0.9
            init_time = ct
        print(f'{len(closedSet) / grad.size:.2f}%, {min_hv:.1f}, {(ct-timer)/ 1_000_000:.2f}, {weight}, {(ct-init_time)/ 1_000_000_000:.1f}')
        timer = ct
        if current == goal: 
            return reconstruct_path(cameFrom, current)
        
        openSet.remove(current)

        if current in closedSet:
            continue
        closedSet.add(current)
        
        for dx, dy in directions:
                if (dx, dy) == (0,0):
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in gScore.keys():
                    gScore[neighbor] = np.inf
                # d is edge weight
                tentative_gScore = gScore[current] + d(current, neighbor, weight * STEP)
                
                if tentative_gScore < gScore[neighbor]: # must act as infinity
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    hv = h(neighbor, goal)
                    if hv < min_hv:
                        min_hv = hv

                    #print(hv, neighbor)
                    fScore[neighbor] = tentative_gScore + hv * 0.01
                    if neighbor not in openSet:
                        #print("appended!", gScore[neighbor], old_gscore)
                        openSet.add(neighbor)

    return 



with rasterio.open("output.tin.tif") as src:
    transform = src.transform
    start = (transform.c, transform.f)
    bounds = src.bounds

    size = (transform.a, transform.e)
    print(start, size)
    elevation_data = src.read(1)

nodata_value = src.nodata
if nodata_value is not None:
    elevation_data = np.where(elevation_data == nodata_value, np.nan, elevation_data)

fig, ax = plt.subplots(1,1,figsize=(10,8))
gx, gy = np.gradient(elevation_data, size[0], size[1])
gx, gy = ndimage.gaussian_filter(gx, sigma=1.0), ndimage.gaussian_filter(gy, sigma=1.0)
grad = np.array([gx,gy]) # arr of vectors
pg = np.stack(grad[::-1], axis=-1)
grad = pg
#grad -= np.nanmin(grad)
# picture = mpimg.imread('vector_map.png')
picture = Image.open('vector_map.png').convert('L')
mask = np.array(picture)
mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
#grad = cv2.GaussianBlur(grad.astype(np.float32), (9, 9), 0)
mask_resized = cv2.resize(mask, (grad.shape[1], grad.shape[0]), interpolation=cv2.INTER_CUBIC)
mask_bool = mask_resized > 128
grad = np.where(mask_bool[..., np.newaxis], grad, [-10000,-10000])
# grad = np.where(mask_resized < 128, grad, 0.00001)
#extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
extent = [0, grad.shape[0], 0, grad.shape[1]]
#image = ax.imshow(elevation_data, extent = extent, norm=LogNorm(), origin='upper')
image = ax.imshow(np.linalg.norm(pg, axis=-1), extent = extent, norm=LogNorm(), origin='upper')
# ax.imshow(picture, alpha=0.5, interpolation='nearest', extent=extent)

best_path = np.array(a_star((1600,1700),(1320,780) , 2000))
x_coords = best_path[:,0] / grad.shape[0] * (extent[1] - extent[0]) + extent[0]
y_coords = best_path[:,1] / grad.shape[1] * (extent[3] - extent[2]) + extent[2]
ax.scatter(x_coords, y_coords, s=50, marker='o')

x = np.linspace(0, extent[1]-1, 500, dtype=np.int32)
y = np.linspace(0, extent[3]-1, 500, dtype=np.int32)
X, Y = np.meshgrid(y,x)
grad = np.where(mask_bool[..., np.newaxis], grad, [-50,-50])
print(X,Y)
j = pg[(Y,X)][::-1] * 30
ax.quiver(X * 1.034,Y * 0.97, j[..., 0], j[..., 1])

cbar = fig.colorbar(image, ax=ax)
ax.set_aspect("equal")

plt.show()
