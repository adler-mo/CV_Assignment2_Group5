# TUWIEN - CV: Task3 - Scene recognition using Bag of Visual Words
# Group 5
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
    sift = cv2.SIFT_create()
    all_descriptors: List[np.ndarray] = []

    for img in images:
        #OpenCV-friendly dtype
        if img.dtype != np.float32:
            img = img.astype(np.float32)

    
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        h, w = img_u8.shape[:2]

        # Build dense keypoint grid
        keypoints = []
        # Starting at stepsize//2 so points are not exactly on the border
        for y in range(stepsize // 2, h, stepsize):
            for x in range(stepsize // 2, w, stepsize):
                keypoints.append(cv2.KeyPoint(float(x), float(y), float(stepsize)))

        # Computing SIFT descriptors at provided keypoints
        _, desc = sift.compute(img_u8, keypoints)

        # Handling cases where no descriptors were produced
        if desc is None or len(desc) == 0:
            all_descriptors.append(np.zeros((0, 128), dtype=np.float32))
            continue

        desc = desc.astype(np.float32)

        # random subsampling per image
        if num_samples is not None and desc.shape[0] > num_samples:
            idx = np.random.choice(desc.shape[0], size=num_samples, replace=False)
            desc = desc[idx]

        all_descriptors.append(desc)
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
    from sklearn.metrics import pairwise_distances

    vocab_size = centroids.shape[0]
    histograms = []

    for desc in dense_feat:
        # Initializing histogram for this image
        hist = np.zeros(vocab_size, dtype=np.float32)

        # Skipping images with no descriptors
        if desc is None or desc.shape[0] == 0:
            histograms.append(hist)
            continue

        # Computing distances: [num_descriptors x vocab_size]
        dists = pairwise_distances(desc, centroids, metric="euclidean")

        # Assigning each descriptor to nearest centroid
        nearest = np.argmin(dists, axis=1)

        # Counting occurrences
        for idx in nearest:
            hist[idx] += 1

        histograms.append(hist)
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
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import normalize

    vocab_size, d = centroids.shape  # Nc, 128
    image_descriptors = []

    for desc in dense_feat:
        # Handling empty descriptor sets
        if desc is None or desc.shape[0] == 0:
            v = np.zeros((vocab_size * d,), dtype=np.float32)
            image_descriptors.append(v)
            continue

        # Assigning each descriptor to nearest centroid
        dists = pairwise_distances(desc, centroids, metric="euclidean")  # [N x Nc]
        nearest = np.argmin(dists, axis=1)                               # [N]

        # Accumulating residuals per centroid: vk = sum_{i assigned to k} (xi - ck)
        V = np.zeros((vocab_size, d), dtype=np.float32)
        for i, k in enumerate(nearest):
            V[k] += (desc[i] - centroids[k])

        # Flattening to 1D VLAD vector [Nc*128]
        v = V.reshape(-1)

        # L2 normalize (sklearn expects 2D)
        v = normalize(v.reshape(1, -1), norm="l2")[0].astype(np.float32)

        image_descriptors.append(v)
    # student_code end
    

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return image_descriptors