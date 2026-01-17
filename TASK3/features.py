# TUWIEN - CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
import sklearn
import sklearn.metrics.pairwise as sklearn_pairwise
import cv2
import numpy as np
import random
import time


def extract_dsift(images: List[np.ndarray], stepsize: int, num_samples: int = None) -> List[np.ndarray]:
    """
    Extracts dense feature points on a regular grid with 'stepsize' and optionally returns
    'num_samples' random samples per image. If 'num_samples' is not provided, it takes all
    features extracted with the given 'stepsize'. SIFT.compute has the argument "keypoints",
    which should be set to a list of keypoints for each square.
    
    Args:
    - images (List[np.ndarray]): List of images to extract dense SIFT features [num_of_images x n x m] - float
    - stepsize (int): Grid spacing, step size in x and y direction.
    - num_samples (int, optional): Random number of samples per image.

    Returns:
    - List[np.ndarray]: SIFT descriptors for each image [number_of_images x num_samples x 128] - float
    """
    tic = time.perf_counter()

    # student_code start
    raise NotImplementedError("TO DO in features.py")
    # student_code end

    toc = time.perf_counter()
    print("DSIFT Extraction:", toc - tic, " seconds")

    # all_descriptors : list sift descriptors per image [number_of_images x num_samples x 128] - float
    return all_descriptors


def count_visual_words(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generates a histogram of word occurrences per image.
    Utilizes sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
    to the nearest centroids and counts the occurrences of each centroid. The histogram
    should be as long as the vocabulary size (number of centroids).

    Args:
    - dense_feat (List[np.ndarray]): List of SIFT descriptors per image [number_of_images x num_samples x 128] - float
    - centroids (List[np.ndarray]): Centroids of clusters [vocabulary_size x 128]

    Returns:
    - List[np.ndarray]: List of histograms per image [number_of_images x vocabulary_size]
    """
    tic = time.perf_counter()

    # student_code start
    raise NotImplementedError("TO DO in features.py")
    # student_code end

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return histograms


def calculate_vlad_descriptors(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generate a histogram of word occurence per image
     Use sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
     to the nearest centroids and calculate for each word the residual to the nearest centroid
     The final feature vector should be as long as the vocabulary size (number of centroids) x feature dimension
     L2-normalize the final descriptors via sklearn.preprocessing.normalize.
     
    Args:
    - dense_feat : list sift descriptors per image [number_of_images x num_samples x 128] - float
    - centroids : centroids of clusters [vocabulary_size x 128]

    Returns:
    - List[np.ndarray]: List of histograms per image [number_of_images x (vocabulary_size x feature dimension)]
    """
    tic = time.perf_counter()

    
    # student_code start
    raise NotImplementedError("TO DO in features.py")
    # student_code end
    

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return image_descriptors