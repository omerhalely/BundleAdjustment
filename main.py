import os
import pykitti
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Frame import Frame
from MatchingPoints import MatchingPoints


def unpack_params(dataset):
    lons = [i.packet.lon for i in dataset.oxts]
    lats = [i.packet.lat for i in dataset.oxts]
    alts = [i.packet.alt for i in dataset.oxts]
    num_sats = [i.packet.numsats for i in dataset.oxts]
    pitches = [i.packet.pitch for i in dataset.oxts]
    rolls = [i.packet.roll for i in dataset.oxts]
    yaws = [i.packet.yaw for i in dataset.oxts]
    times = dataset.timestamps
    accs = [i.packet.pos_accuracy for i in dataset.oxts]
    return lons, lats, alts, num_sats, accs, pitches, rolls, yaws, times


def add_next_frame(dataset : pykitti.raw, next_frame : int, orb : cv2.ORB, frames : list, kp : list, des : list):
    frames.pop(0)
    kp.pop(0)
    des.pop(0)

    frames.append(np.array(dataset.get_cam0(next_frame)))
    keypoints, descriptors = orb.detectAndCompute(frames[-1], None)
    kp.append(keypoints)
    des.append(descriptors)

    return frames, kp, des


def reprojection_error(u, v, X, P):
    X = np.reshape(np.concatenate((X, np.array([1]))), (4, 1))
    reprojection = P @ X
    reprojection = reprojection[:-1, :] / reprojection[-1, 0]

    error = np.sqrt((u - reprojection[0, 0]) ** 2 + (v - reprojection[1, 0]) ** 2)
    return error


def depth_estimation(frames, matching_points, C, K):
    depth_image = np.zeros_like(frames[0].frame)
    reference_indices = list(matching_points.keys())
    reference_points = np.array([frames[0].features[reference_indices[i]].pt for i in range(len(reference_indices))])
    projection_matrices = [C @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)]
    points = [reference_points]
    for k in range(1, K):
        points_k = np.array([frames[k].features[matching_points[reference_indices[i]][k - 1]].pt for i in range(len(reference_indices))])
        points.append(points_k)

        F, mask = cv2.findFundamentalMat(points_k, reference_points, cv2.FM_RANSAC, 0.1, 0.99)
        E = C.T @ F @ C
        _, R, t, _ = cv2.recoverPose(E, points_k, reference_points, cameraMatrix=C)
        P = C @ np.concatenate((R, t), axis=1)
        projection_matrices.append(P)

    for i in range(len(projection_matrices)):
        frames[i].set_projection_matrix(projection_matrices[i], C)

    points = np.array(points)
    for i in range(points.shape[1]):
        A = None
        for (j, point) in enumerate(points[:, i, :]):
            P = projection_matrices[j]
            P1 = P[0, :]
            P2 = P[1, :]
            P3 = P[2, :]
            u, v = point
            if A is None:
                A = np.array([
                    u * P3 - P1,
                    v * P3 - P2,
                ])
            else:
                new_rows = np.array([
                    u * P3 - P1,
                    v * P3 - P2,
                ])
                A = np.concatenate((A, new_rows), axis=0)

        _, _, Vt = np.linalg.svd(A.T @ A)
        X = Vt[-1]
        X = X[:-1] / X[-1]
        depth_value = np.linalg.norm(X)
        depth_image[int(points[0, 0, 1]), int(points[0, 0, 0])] = depth_value

    return R, t


def bundle_adjustment(dataset, K):
    orb = cv2.ORB_create(5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    C = dataset.calib.P_rect_00[:, :3]

    frames = []
    matching_points = MatchingPoints()

    rotations = [np.eye(3)]
    translations = [np.zeros((3, 1), dtype=np.float32)]
    trajectory = np.array([]).reshape(0, 2)

    for i in tqdm(range(len(dataset))):
        if len(frames) < K:
            new_frame = Frame(np.array(dataset.get_cam0(i)))
            new_frame.process_frame(orb)
            frames.append(new_frame)
            continue

        for frame in frames[1:]:
            matching_points.add(frames[0].match(frame, bf))
        matching_points.unique()
        R, t = depth_estimation(frames, matching_points.matching_points, C, K)

        translations.append(translations[-1] + rotations[-1] @ t)
        rotations.append(rotations[-1] @ R)
        trajectory = np.concatenate((trajectory, np.array(
            [[translations[-1][0, 0], translations[-1][2, 0]]]
        )))

        frames.pop(0)
        new_frame = Frame(np.array(dataset.get_cam0(i)))
        new_frame.process_frame(orb)
        frames.append(new_frame)

        matching_points.reset()

    poses = np.array([]).reshape(0, 2)
    rotations = [np.eye(3)]
    translations = [np.zeros((3, 1), dtype=np.float32)]
    T_cam0_imu = dataset.calib.T_cam0_imu
    for oxts in dataset.oxts:
        T_w_imu = oxts.T_w_imu
        T_w_cam0 = T_w_imu @ np.linalg.inv(T_cam0_imu)

        R = T_w_cam0[:3, :3]
        t = np.reshape(T_w_cam0[:3, 3], (3, 1))

        translations.append(translations[-1] + rotations[-1] @ t)
        rotations.append(R @ rotations[-1])

        poses = np.concatenate((poses, np.array(
            [[translations[-1][0, 0], translations[-1][2, 0]]]
        )))

    plt.plot(poses[:, 0], poses[:, 1])
    plt.plot((855 / 34.4) * trajectory[:, 0], (1272 / 50.95) * trajectory[:, 1])
    plt.show()


if __name__ == "__main__":
    basedir = os.path.join(os.getcwd(), "data")
    date = '2011_09_26'
    drive = '0002'

    dataset = pykitti.raw(basedir, date, drive)

    lons, lats, alts, num_satsa, accs, pitches, rolls, yaws, times = unpack_params(dataset)

    K = 3
    bundle_adjustment(
        dataset=dataset,
        K=K
    )
