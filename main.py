import os
import pykitti
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def depth_estimation(frame, height, width, d, keypoints, C):
    for key in d:
        matching_points = np.array(list(d[key].items()))
        # pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts1 = keypoints[0][matching_points[:, 0], :]
        pts2 = keypoints[key + 1][matching_points[:, 1], :]

        F, mask = cv2.findFundamentalMat(pts2, pts1, cv2.FM_RANSAC, 0.1, 0.99)
    # pts1 = pts1[mask[:, 0] == 1]
    # pts2 = pts2[mask[:, 0] == 1]
    E = C.T @ F @ C

    _, R, t, _ = cv2.recoverPose(E, pts2, pts1, cameraMatrix=C)

    P = C @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    P_prime = C @ np.concatenate((R, t), axis=1)

    P1 = P[0, :]
    P2 = P[1, :]
    P3 = P[2, :]
    P_prime1 = P_prime[0, :]
    P_prime2 = P_prime[1, :]
    P_prime3 = P_prime[2, :]
    errors = []
    depth = np.zeros((height, width), dtype=np.float32)
    for (pt1, pt2) in zip(pts1, pts2):
        u1, v1 = pt1
        u2, v2 = pt2

        A = np.array([
            u1 * P3 - P1,
            v1 * P3 - P2,
            u2 * P_prime3 - P_prime1,
            v2 * P_prime3 - P_prime2
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:-1] / X[-1]

        # eig_values, eig_vectors = np.linalg.eig(A)
        # X = eig_vectors[:, np.argmin(np.abs(eig_values))]
        # X = X[:-1] / X[-1]

        error = reprojection_error(u1, v1, X, P)

        depth_value = np.linalg.norm(X)
        if depth_value > 80:
            continue
        color = 255 * depth_value / 80
        cv2.circle(frame, (int(u1), int(v1)), 3, color, 2)
        depth[int(v1), int(u1)] = depth_value
        errors.append(error)

    print(np.mean(np.array(errors)))
    depth[depth > 80] = 0

    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(frame)
    axis[1].imshow(depth)
    plt.show()
    return R, t


def dictionary_intersection(d):
    for key in d:
        d[key] = {k: d[key][k] for k in d[key] if k in d[(key + 1) % len(d)]}
    return d


def bundle_adjustment(dataset, K):
    orb = cv2.ORB_create(5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    C = dataset.calib.P_rect_00[:, :3]

    frames = [np.array(dataset.get_cam0(i)) for i in range(K)]
    height, width = frames[0].shape

    features = [orb.detectAndCompute(frames[i], None) for i in range(len(frames))]
    kp = [features[i][0] for i in range(len(features))]
    des = [features[i][1] for i in range(len(features))]

    rotations = [np.eye(3)]
    translations = [np.zeros((3, 1), dtype=np.float32)]
    trajectory = np.array([]).reshape(0, 2)
    matching_points = {}
    for i in range(K, len(dataset)):
        reference_keypoints = kp[0]
        reference_descriptor = des[0]
        j = 0
        for descriptor in des[1:]:
            matches = bf.knnMatch(reference_descriptor, descriptor, k=2)

            matching_points[j] = {}
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    matching_points[j][m.queryIdx] = m.trainIdx


            # R, t = depth_estimation(frames[0], height, width, pts1, pts2, C)
            # translations.append(translations[-1] + rotations[-1] @ t)
            # rotations.append(rotations[-1] @ R)
            # trajectory = np.concatenate((trajectory, np.array(
            #     [[translations[-1][0, 0], translations[-1][2, 0]]]
            # )))

            j += 1

        # matching_points[1] = {k : matching_points[1][k] for k in matching_points[1] if k in matching_points[2]}
        # matching_points[2] = {k : matching_points[2][k] for k in matching_points[2] if k in matching_points[1]}
        matching_points_intersection = dictionary_intersection(matching_points)
        depth_estimation(frames[0], height, width, matching_points_intersection, kp, C)
        frames, kp, des = add_next_frame(
            dataset=dataset,
            next_frame=i,
            orb=orb,
            frames=frames,
            kp=kp,
            des=des
        )

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

    fig, axis = plt.subplots(1, 2)
    axis[0].plot(poses[:, 0], poses[:, 1])
    axis[1].plot(trajectory[:, 0], trajectory[:, 1])
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
