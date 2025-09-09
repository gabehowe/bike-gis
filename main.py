"""
Finds the optimal path between two points while factoring in the work required to move up hills in a GeoTIFF file with the a* algorithm. 
"""
__author__ = "gabri"

import rasterio
import rasterio.warp
import numpy as np
import matplotlib
matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage
import time
import types
from types import List, Tuple
import cv2
from PIL import Image


def reconstruct_path(cameFrom: dict, current: tuple):
    """
    Retraces the path to back from the current node.
    Args:
        current: The position from which to trace back.
        cameFrom: The dict which stores backreferences to the previous node in the path.

    Returns: 
        The most optimal path from the current node to the start.
    """
    total_path = [current]
    while current in cameFrom:
        current = cameFrom[current]
        total_path.append(current)
    return total_path[::-1]

def d(current: tuple, point: tuple, weight: float):
    """
    The edge weight function for a*. Determines move weight based on work against normal force required to move, i.e., will punish moving up hill and reward moving down hill.

    Args:
        current: The current position of the a* algorithm.
        point: The point to find the edge weight to.
        weight: The slope weighting of the function.

    Returns:
        float: The score of the edge -- higher values are worse.

    Note:
        Accessing grad in this function makes this susceptible to side effects.
    """
    if not (point[0] in range(0,grad.shape[0]) and point[1] in range(0, grad.shape[1])):
        return 1000000000
    if np.isnan(grad[point][0]):
        return 3000 * weight # should be hard to move into buildings

    vector = np.array((point[0] - current[0], point[1] - current[1]))
    # The normal force should contribute on downward slopes and hinder on upward slopes.
    # Units are m/s^2 * kg * (m) = work
    normal_force = vector @ -grad[point] * 9.81 * 80 * weight
    required_horizontal_force = 80 * np.linalg.norm(vector) # = work (force time distance)
    rv =  required_horizontal_force + normal_force 
    return max( rv, 0)

def h(it, goal):
    return np.hypot(*(goal[0] - it[0], goal[1] - it[1]))

def a_star(start: tuple, goal: tuple, weight: float,step=8, acceptable_time=15) -> Tuple[List[tuple], List[float]]:
    """
    Implementation of the a* pathing algorithm with reducing weight after exceeding a time limit.

    Note:
        Also see the wikipedia page for a*, which has this piece of code almost verbatim.

    Args: 
        start: The position from which to start.
        goal: The position at which to arrive.
        weight: The multiplier on the slope formula. Higher numbers result in distance weighing less and avoiding slopes weighing more.
        step: The distance to move each iteration. Higher numbers result in a quicker completion but less accuracy as a result of decreased sampling.
        acceptable_time: The time in seconds before the weight will be decreased in an attempt to curb extremely long wait times.

    Returns:
        A tuple of the points for the optimal path and their expenses.
        
    """
    # ndarrays are tuples
    openSet = set([start])
    closedSet = set()
    cameFrom = {}
    gScore = {} # default value of infinity
    gScore[start] = 0
    min_hv = np.inf
    tScore = {} 
    tScore[start] = 0

    fScore = {} # default value of infinity
    fScore[start] = h(start, goal)
    hv = np.inf
    directions = [(x*step,y*step) for x in range(-1,2) for y in range(-1,2)]
    init_time = time.perf_counter_ns()
    timer = init_time
    while len(openSet) > 0:

        if min_hv <= 30:
            rp = reconstruct_path(cameFrom, current)

            return rp, [tScore[i] for i in rp]
        current = min(openSet, key=lambda x: fScore[x])
        ct = time.perf_counter_ns()
        if (ct-init_time) / 1_000_000_000 > acceptable_time:
            weight *= 0.9
            init_time = ct
        #print(f'{len(closedSet) / grad.size:.2f}%, {min_hv:.1f}, {(ct-timer)/ 1_000_000:.2f}, {weight}, {(ct-init_time)/ 1_000_000_000:.1f}')
        timer = ct
        if current == goal: 
            rp = reconstruct_path(cameFrom, current)

            return rp, [tScore[i] for i in rp]
        
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
                calc_gs =  d(current, neighbor, weight)
                tentative_gScore = gScore[current] + calc_gs
                
                if tentative_gScore < gScore[neighbor]: # must act as infinity
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    tScore[neighbor] = calc_gs
                    hv = h(neighbor, goal)
                    if hv < min_hv:
                        min_hv = hv

                    #print(hv, neighbor)
                    fScore[neighbor] = tentative_gScore + hv
                    if neighbor not in openSet:
                        #print("appended!", gScore[neighbor], old_gscore)
                        openSet.add(neighbor)

    return 


def convert_from_epsg4326(p: tuple, crs_start, crs) -> np.ndarray:
    """
    Converts an epsg4326 tuple into our image space.

    Args:
        p: An epsg4326 tuple to convert.
        crs_start: the top leftmost coordinate.
        crs: the coordinate system information.

    Returns:
        `ndarray`: An int64 (2,) numpy array with the coordinates for indexing into the elevation data.
    """

    return np.astype((np.array(rasterio.warp.transform("EPSG:4326", crs, [p[0]], [p[1]])).T - crs_start)[0] * (1, -1), np.int64)

def find_lowest_energy_path(sp: tuple, fp: tuple, weight: float):
    """
    Finds the lowest energy path between epsg4326 coordinates (long, lat) on the USC campus with the provided output.tin.tif. Could be generalized.

    Args:
        sp: An epsg4326 tuple from which to start.
        fp: An epsg4326 tuple at which to end.
        weight: The effect of slope on the pathing algorithm.
    """
    with rasterio.open("output.tin.tif") as src:
        print("file loaded")
        transform = src.transform
        start = (transform.c, transform.f)
        crs_start =  transform * (0,0)
        elevation_data = src.read(1)

        nodata_value = src.nodata
        if nodata_value is not None:
            elevation_data = np.where(elevation_data == nodata_value, np.nan, elevation_data)

    fig, ax = plt.subplots(1,1,figsize=(10,8))

    ax.set_aspect("equal")
    # get gradient (slope)
    gx, gy = np.gradient(elevation_data)
    gx, gy = ndimage.gaussian_filter((gx,gy), sigma=1.1)
    grad = np.array([gx,gy]) # arr of vectors
    grad = np.stack(grad[::-1], axis=-1)

    # try to remove buildings and keep roads
    picture = Image.open('vector_map.png').convert('L')
    mask = np.array(picture)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
    mask_resized = cv2.resize(mask, (grad.shape[1], grad.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask_bool = mask_resized > 128
    grad = np.where(mask_bool[..., np.newaxis], grad, [np.nan, np.nan])
    # Get start and end coordinates from input coordinates
    start = convert_from_epsg4326(sp, crs_start, src.crs)
    finish = convert_from_epsg4326(fp, crs_start, src.crs)

    path, t_score = a_star(tuple(np.flip(finish)),tuple(np.flip(start)), weight)
    best_path = np.flip(np.array(path))
    # Get real min and max for framing
    dmin_x, dmin_y = best_path.min(axis=0)
    dmax_x, dmax_y = best_path.max(axis=0)
    # Expand limits to make framing better
    FRAMING_EXPANSION = 0.15
    min_x, min_y = np.astype(np.array((dmin_x, dmin_y)) * (1-FRAMING_EXPANSION), np.int64)
    max_x, max_y = np.astype(np.array((dmax_x, dmax_y)) * (1+FRAMING_EXPANSION), np.int64)


    # Magnitude of the gradient at every point to color.
    data = np.linalg.norm(grad[min_y:max_y, min_x:max_x], axis=-1)
    image = ax.imshow(data, norm=LogNorm())

    # Reframe the path. 
    x_coords = (best_path[:,0])-min_x
    y_coords = (best_path[:,1])-min_y
    pts = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)

    # Create arrow segments for colors -- this could be done more succinctly.
    segments = np.concatenate([pts[:-1], pts[1:]],axis=1)

    # Create direction vector for arrows.
    start_points  = segments[:, 0]
    end_points = segments[:, 1]
    diff = (end_points - start_points)
    direction = diff/np.linalg.norm(diff) * 12


    ax.quiver(start_points[:, 0], start_points[:, 1], direction[:, 0], direction[:, 1], 
              np.nan_to_num(t_score[:-1], 1) - min(t_score) + 0.1,
              cmap='plasma',
              norm=LogNorm(vmin=1e-8),
              angles='xy', 
              scale_units='xy', 
              scale=0.1)

    plt.show()
    # Trip info
    print(f"Len: {np.sum(np.linalg.norm(segments[:, :1] - segments[:, 1:], axis=-1)):.1f} m")
    print(f'Effort (80 kg * force * distance): {np.sum(t_score):.1f} J')
    print(f'Kilocalories: {np.sum(t_score)/4184}')
    print(f'Elevation Change: {(elevation_data[start] - elevation_data[finish]) * 6}')



start = (-81.0255, 33.9981)
finish = (-81.0345025, 33.9932168)
weight = 1
find_lowest_energy_path(start,finish, weight)
