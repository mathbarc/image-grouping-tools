import logging
from numpy import argsort
from tqdm import tqdm
from torch import mean, stack
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

logger = logging.getLogger()

def get_referece_vector(ref_images: list, data: dict):
    ref_tensors = []
    for p in data['paths']:
        if ref_images[0].split('/')[-1] in str(p):
            ref_tensors.append(data['features'][data['paths'].index(p)])
    if len(ref_tensors) > 1:
        return mean(stack(ref_tensors, dim=0), dim=0)
    return ref_tensors[0]

        

def get_distances(data: dict, reference_images: list, dist_type: str) -> list:
    distances = []
    ref_img_features = get_referece_vector(reference_images, data)
    logger.info("Computating distances...")
    for i in tqdm(range(data["features"].size(dim=0))):
        distances.append(functions[dist_type](data["features"][i], ref_img_features))
    sorted_idx = argsort(distances, axis=0)
    return distances, sorted_idx
