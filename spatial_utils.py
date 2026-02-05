# spatial_utils.py
import math
import numpy as np
import torch


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def build_graph_using_normalized_boxes(bbox, label_num=11, distance_threshold=0.5):
    num_box = bbox.shape[0]
    adj_matrix = np.zeros((num_box, num_box), dtype=np.int8)

    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    image_diag = math.sqrt(2)

    for i in range(num_box):
        if sum(bbox[i]) == 0: continue
        adj_matrix[i, i] = 12  # self relation
        for j in range(i + 1, num_box):
            if sum(bbox[j]) == 0: continue

            if xmin[i] < xmin[j] and xmax[i] > xmax[j] and ymin[i] < ymin[j] and ymax[i] > ymax[j]:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 2
            elif xmin[j] < xmin[i] and xmax[j] > xmax[i] and ymin[j] < ymin[i] and ymax[j] > ymax[i]:
                adj_matrix[i, j] = 2
                adj_matrix[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbox[i], bbox[j])
                if ioU >= 0.5:
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
                else:
                    y_diff = center_y[i] - center_y[j]
                    x_diff = center_x[i] - center_x[j]
                    diag = math.sqrt((y_diff) ** 2 + (x_diff) ** 2)

                    if diag < distance_threshold * image_diag:
                        if diag == 0:
                            adj_matrix[i, j] = 3
                            adj_matrix[j, i] = 3
                        else:
                            sin_ij = y_diff / diag
                            cos_ij = x_diff / diag
                            if sin_ij >= 0 and cos_ij >= 0:
                                label_i = np.arcsin(sin_ij); label_j = math.pi + label_i
                            elif sin_ij < 0 and cos_ij >= 0:
                                label_i = np.arcsin(sin_ij) + 2 * math.pi; label_j = label_i - math.pi
                            elif sin_ij >= 0 and cos_ij < 0:
                                label_i = np.arccos(cos_ij); label_j = label_i + math.pi
                            else:
                                label_i = 2 * math.pi - np.arccos(cos_ij); label_j = label_i - math.pi

                            adj_matrix[i, j] = int(np.ceil(label_i / (math.pi / 4))) + 3
                            adj_matrix[j, i] = int(np.ceil(label_j / (math.pi / 4))) + 3
    return adj_matrix