import numpy as np
import torch


class Frame:
    def __init__(self, frame):
        self.frame = frame
        self.height, self.width = self.frame.shape
        self.features = None
        self.descriptor = None
        self.P = None
        self.R = None
        self.t = None

    def process_frame(self, orb):
        self.features, self.descriptor = orb.detectAndCompute(self.frame, None)

    def match(self, other, bf):
        matches = bf.knnMatch(self.descriptor, other.descriptor, k=2)
        matching_points = {}
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                matching_points[m.queryIdx] = m.trainIdx

        return matching_points

    def set_projection_matrix(self, P, C):
        self.P = P
        R_t = np.linalg.inv(C) @ P
        self.R = R_t[:, :-1]
        self.t = np.reshape(R_t[:, -1], (3, 1))

    def to_tensor(self, device):
        self.P = torch.from_numpy(self.P).requires_grad_(True).to(device)
        self.R = torch.from_numpy(self.R).requires_grad_(True).to(device)
        self.t = torch.from_numpy(self.t).requires_grad_(True).to(device)
