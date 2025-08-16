import os
import pykitti
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

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


def reprojection_error(P, X, x, height, width, device):
    ones = torch.ones(X.shape[0], 1, device=device)
    X = torch.concatenate((X, ones), dim=-1)
    reprojection = (P @ X.T).T
    reprojection = torch.reshape(reprojection, (reprojection.shape[0], -1, 3))
    reprojection = reprojection[:, :, :-1] / torch.unsqueeze(reprojection[:, :, -1], dim=-1)
    reprojection[:, :, 0] /= width
    reprojection[:, :, 1] /= height
    reprojection = reprojection.flatten()

    error = torch.sqrt(torch.mean((x - reprojection) ** 2))
    return error


def camera_projection_function(projection_matrix, world_coordinates):
    ones_column = torch.ones(world_coordinates.shape[0], 1, requires_grad=False).to(device)
    homogeneous_coordinates = torch.concatenate((world_coordinates, ones_column), dim=-1)
    width, height = 1200, 375
    camera_projection = (projection_matrix @ homogeneous_coordinates.T).T
    camera_projection = torch.reshape(camera_projection, (camera_projection.shape[0], -1, 3))
    camera_projection = camera_projection / torch.unsqueeze(camera_projection[:, :, -1], dim=-1)
    camera_projection = camera_projection[:, :, :-1]
    camera_projection[:, :, 0] /= width
    camera_projection[:, :, 1] /= height
    camera_projection = torch.flatten(camera_projection)
    return camera_projection


def LMA_optimization(frames, camera_coordinates, world_coordinates, projection_matrix, height, width, device):
    ones_column = torch.ones(world_coordinates.shape[0], 1, requires_grad=False).to(device)
    homogeneous_coordinates = torch.concatenate((world_coordinates, ones_column), dim=-1)

    camera_projection = (projection_matrix @ homogeneous_coordinates.T).T
    camera_projection = torch.reshape(camera_projection, (camera_projection.shape[0], -1, 3))
    camera_projection = camera_projection / torch.unsqueeze(camera_projection[:, :, -1], dim=-1)
    camera_projection = camera_projection[:, :, :-1]
    camera_projection[:, :, 0] /= width
    camera_projection[:, :, 1] /= height
    camera_projection = torch.flatten(camera_projection)

    params = (projection_matrix, world_coordinates)
    J_P, J_X = torch.autograd.functional.jacobian(camera_projection_function, params)
    J_P = torch.flatten(J_P, start_dim=1)
    J_X = torch.flatten(J_X, start_dim=1)
    J = torch.concatenate((J_P, J_X), dim=1)

    lambda0 = 1e-3
    JT_J = J.T @ J
    A = JT_J + lambda0 * torch.diag(torch.diag(JT_J))
    b = camera_coordinates - camera_projection
    delta = torch.linalg.inv(A) @ J.T @ b
    delta_P = delta[:3 * len(frames) * 4]
    delta_X = delta[3 * len(frames) * 4:]

    delta_P = torch.reshape(delta_P, projection_matrix.shape)
    delta_X = torch.reshape(delta_X, world_coordinates.shape)
    return delta_P, delta_X


def depth_estimation(frames, matching_points, C, K, device):
    depth_image = np.zeros_like(frames[0].frame)
    reference_indices = list(matching_points.keys())
    reference_points = np.array([frames[0].features[reference_indices[i]].pt for i in range(len(reference_indices))])
    projection_matrices = [C @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)]
    points = [reference_points]
    ransac_mask = None
    for k in range(1, K):
        points_k = np.array([frames[k].features[matching_points[reference_indices[i]][k - 1]].pt for i in range(len(reference_indices))])
        points.append(points_k)

        F, mask = cv2.findFundamentalMat(points_k, reference_points, cv2.FM_RANSAC, 0.1, 0.99)
        if ransac_mask is None:
            ransac_mask = mask
        else:
            ransac_mask = np.logical_and(ransac_mask, mask)
        E = C.T @ F @ C
        _, R, t, _ = cv2.recoverPose(E, points_k, reference_points, cameraMatrix=C)
        P = C @ np.concatenate((R, t), axis=1)
        projection_matrices.append(P)

    for i in range(len(points)):
        points[i] = points[i][ransac_mask[:, 0] == 1]

    for i in range(len(projection_matrices)):
        frames[i].set_projection_matrix(projection_matrices[i], C)

    points = np.array(points)
    world_coordinates = np.zeros((points.shape[1], 3))
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
        X = X / X[-1]

        world_coordinates[i, :] = X[:-1]

        X = X[:-1]
        depth_value = np.linalg.norm(X)
        depth_image[int(points[0, 0, 1]), int(points[0, 0, 0])] = depth_value

    height, width = frames[0].height, frames[0].width
    camera_coordinates = torch.from_numpy(points).requires_grad_(False).to(device)
    camera_coordinates = torch.permute(camera_coordinates, (1, 0, 2))
    camera_coordinates[:, :, 0] /= width
    camera_coordinates[:, :, 1] /= height
    camera_coordinates = torch.flatten(camera_coordinates)
    world_coordinates = torch.from_numpy(world_coordinates).requires_grad_(True).to(device)

    projection_matrix = None
    for i in range(len(frames)):
        frames[i].to_tensor(device)
        if projection_matrix is None:
            projection_matrix = frames[i].P
        else:
            projection_matrix = torch.concatenate((projection_matrix, frames[i].P), dim=0)
    projection_matrix.to(device).requires_grad_(True)

    max_iter = 2
    # error = reprojection_error(projection_matrix, world_coordinates, camera_coordinates, height, width, device)
    for i in range(max_iter):
        delta_P, delta_X = LMA_optimization(frames, camera_coordinates, world_coordinates, projection_matrix, height, width, device)

        projection_matrix += delta_P
        world_coordinates += delta_X
        # error = reprojection_error(projection_matrix, world_coordinates, camera_coordinates, height, width, device)

    C = torch.from_numpy(C).to(device)
    P = projection_matrix.unfold(0, 3, 3)
    R_t = P @ torch.linalg.inv(C.T)
    R_t = torch.transpose(R_t, dim0=1, dim1=-1)
    return R_t[1].cpu().detach().numpy()


def bundle_adjustment(dataset, K, device):
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
        R_t = depth_estimation(frames, matching_points.matching_points, C, K, device)

        R = R_t[:, :-1]
        t = np.reshape(R_t[:, -1], (R_t.shape[0], 1))

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
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    basedir = os.path.join(os.getcwd(), "data")
    date = '2011_09_26'
    drive = '0002'

    dataset = pykitti.raw(basedir, date, drive)

    lons, lats, alts, num_satsa, accs, pitches, rolls, yaws, times = unpack_params(dataset)

    K = 3
    bundle_adjustment(
        dataset=dataset,
        K=K,
        device=device
    )
