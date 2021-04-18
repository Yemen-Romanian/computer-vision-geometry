import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from scipy.io import loadmat
from itertools import combinations


def find_matches(im1, im2, nfeatures=100):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f'Found {len(good_matches)}')
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)    
    return src_points, dst_points


def compute_transfer_error(H, src, dst):
    _U = H.dot(src.T).T
    _U[:, 0] /= _U[:, 2]
    _U[:, 1] /= _U[:, 2]
    return np.linalg.norm(_U - dst)


def find_homography(src_points, dst_points):
    def cross_matrix(a):
        result = np.array([[0, -a[2], a[1]],
                           [a[2], 0, -a[0]],
                           [-a[1], a[0], 0]])
        return result

    def homography(src, dst):
        "Homography estimation from 4 points"
        M = []
        for s, d in zip(src, dst):
            mat = np.kron(cross_matrix(d), s)
            M.append(mat)
        M = np.array(M).reshape((-1, 9))
        H = null_space(M)
        if H.size != 9:
            H = H[:, 0]
        H = H.reshape((3, 3))
        H /= H[2, 2]
        return H

    if (len(src_points) < 4):
        print('Not enough points for homography estimation.' +
                f'Need at least 4, provided {len(src_points)}')
        return None, None

    # convert image coordinates to homogeneous vectors
    src_points = np.hstack([src_points, np.ones((len(src_points), 1))])
    dst_points = np.hstack([dst_points, np.ones((len(dst_points), 1))])
    
    if (len(src_points) == 4):
        return homography(src_points, dst_points), 0

    else:
        min_error = np.inf
        for indices in combinations(np.arange(len(src_points)), 4):
            sel_s, sel_d = src_points[list(indices)], dst_points[list(indices)]
            estimated_H = homography(sel_s, sel_d)
            error = compute_transfer_error(estimated_H, src_points, dst_points)
            if error < min_error:
                min_error = error
                best_H = estimated_H
                print(min_error)

        return best_H, min_error


def apply_homography(src_image, H, dst_image_shape):
    dst_image = np.zeros(dst_image_shape)

    for x in range(src_image.shape[0]):
        for y in range(src_image.shape[1]):
            dst_image[int(H[0, :].dot([x, y, 1]) / H[2, :].dot([x, y, 1])), int(H[1, :].dot([x, y, 1]) / H[2, :].dot([x, y, 1])), :] = src_image[x, y, :]

    return dst_image
