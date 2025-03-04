import random
from .distance_functions import (euclidian_distance, 
                                 cosine_distance, 
                                 minkowski_distance, 
                                 mahalanobis_distance)

functions = {
    "euclidian": euclidian_distance,
    "cosine": cosine_distance,
    "minkowski": minkowski_distance(5),
    "mahalanobis": mahalanobis_distance
}


def get_n_diverse_images(points: dict, n: int, dist_type: str):
    centers_idx = apply_k_center_algorithm(points, n, dist_type)
    return centers_idx
    

def apply_k_center_algorithm(features: list, k: int, distance_function: str):
    centers_idx = random.choice(range(len(features)))
    centers = [centers_idx]
    for _ in range(k-1):
        max_distance = 0
        farthest_point_index = None
        for i, point in enumerate(features):
            min_distance_to_center = float('inf')
            for center_index in centers:
                distance = functions[distance_function](point, features[center_index]) 
                min_distance_to_center = min(min_distance_to_center, distance)
            if min_distance_to_center > max_distance:
                max_distance = min_distance_to_center
                farthest_point_index = i
        centers.append(farthest_point_index)
    return centers


def apply_k_means_plus_plus_algorithm():
    pass