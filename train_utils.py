import copy
import operator
import pathlib
import pickle
import random
from typing import Any, Union
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def Transfer_quat(quats):
    """compute R (output) such that v(inertial)=R v(body)"""
    q0 = quats[0]
    q1 = quats[1]
    q2 = quats[2]
    q3 = quats[3]

    return np.array([[np.square(q0) + np.square(q1) - np.square(q2) - np.square(q3), 2 * (q1 * q2 + q0 * q3),
                      2 * (q1 * q3 - q0 * q2)],
                     [2 * (q1 * q2 - q0 * q3), np.square(q0) - np.square(q1) - np.square(q2) - np.square(q3),
                      2 * (q0 * q1 + q2 * q2)],
                     [2 * (q0 * q2 + q1 * q3), 2 * (q2 * q3 - q0 * q1),
                      np.square(q0) - np.square(q1) - np.square(q2) + np.square(q3)]])


def EulerAngles2Quaternions(phi_theta_psi):
    """Input is a Nx3 vector and output is a Nx4 vector"""

    phi = phi_theta_psi[0]
    theta = phi_theta_psi[1]
    psi = phi_theta_psi[2]

    q0 = np.multiply(np.multiply(np.cos(phi / 2), np.cos(theta / 2)), np.cos(psi / 2)) + \
         np.multiply(np.multiply(np.sin(phi / 2), np.sin(theta / 2)), np.sin(psi / 2))
    q1 = np.multiply(np.multiply(np.sin(phi / 2), np.cos(theta / 2)), np.cos(psi / 2)) - \
         np.multiply(np.multiply(np.cos(phi / 2), np.sin(theta / 2)), np.sin(psi / 2))
    q2 = np.multiply(np.multiply(np.cos(phi / 2), np.sin(theta / 2)), np.cos(psi / 2)) + \
         np.multiply(np.multiply(np.sin(phi / 2), np.cos(theta / 2)), np.sin(psi / 2))
    q3 = np.multiply(np.multiply(np.cos(phi / 2), np.cos(theta / 2)), np.sin(psi / 2)) - \
         np.multiply(np.multiply(np.sin(phi / 2), np.sin(theta / 2)), np.cos(psi / 2))

    return np.transpose([q0, q1, q2, q3])


def Quaternions2EulerAngles(quatornian):
    """Input is a Nx3 vector and output is a Nx4 vector"""

    q0 = quatornian[0]
    q1 = quatornian[1]
    q2 = quatornian[2]
    q3 = quatornian[3]

    norm = np.sqrt(np.square(q0) + np.square(q1) + np.square(q2) + np.square(q3))
    q0 = q0 / norm
    q1 = q1 / norm
    q2 = q2 / norm
    q3 = q3 / norm
    phi = np.arctan2(2 * (np.multiply(q0, q1) + np.multiply(q2, q3)),
                     1 - 2 * (np.square(q1) + np.square(q2)))
    theta = np.arcsin(np.clip(2 * (np.multiply(q0, q2) - np.multiply(q1, q3)), a_min=-1.0, a_max=1.0))
    tsi = np.arctan2(2 * (np.multiply(q0, q3) + np.multiply(q1, q2)),
                     1 - 2 * (np.square(q2) + np.square(q3)))
    return np.array([phi, theta, tsi])


def Quaternions2RotationMatrix(quat):
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    matrix = np.array(
        [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
         [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
         [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])
    return matrix


def RotationMatrix2Quaternions(matrix):
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]
    w = np.sqrt(1.0 + r11 + r22 + r33) / 2.0
    x = (r32 - r23) / (4.0 * w)
    y = (r13 - r31) / (4.0 * w)
    z = (r21 - r12) / (4.0 * w)

    return np.array([w, x, y, z])


def RotationMatrix2SixD(matrix):
    first = matrix[:, 0]
    second = matrix[:, 1]
    new_matrix = np.array([first, second]).transpose()

    return new_matrix


def SixD2RotationMatrix(vector):
    first = vector[:, 0]
    second = vector[:, 1]
    x = first / np.linalg.norm(first)
    y = second - np.dot(x, second) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    final = np.array([x, y, z]).transpose()

    return final


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mean = torch.zeros(shape, dtype=torch.float32).to(self.device)
        self.var = torch.ones(shape, dtype=torch.float32).to(self.device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.square() * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean, self.var, self.count = new_mean, new_var, new_count


def save_to_pkl(path: Union[str, pathlib.Path], obj: Any) -> None:
    """
    Save an object to path creating the necessary folders along the way.
    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.
    """
    with open(path, 'wb') as file_handler:
        # Use protocol>=4 to support saving replay buffers >= 4Gb
        # See https://docs.python.org/3/library/pickle.html
        pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pkl(path: Union[str, pathlib.Path]) -> Any:
    """
    Load an object from the path. If a suffix is provided in the path, it will use that suffix.
    If the path does not exist, it will attempt to load using the .pkl suffix.
    """
    with open(path, 'rb') as file_handler:
        return pickle.load(file_handler)


def weights_init(m, init_type='normal'):
    """ Weigh and bias initialization
        Initialize same type nn layers
        e.g. self.actor_target.apply(weights_init)
    """
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.bias.data, 0.0)
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 1e-5)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=1e-5)  # tanh, sigmoid
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')  # relu
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))


class SegmentTree:
    """ Create SegmentTree.
    Segment tree for Proirtized Replay Buffer.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    Attributes:
        capacity (int)
        tree (list)
        operation (function)
    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.
        Args:
            capacity (int)
            operation (function)
            init_value (float)
        """
        assert (
                capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
            self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int):
        """Initialization.
        Args:
            capacity (int)
        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int):
        """Initialization.
        Args:
            capacity (int)
        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


class GaussianNoise:
    """Gaussian Noise.
    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
            self,
            action_dim: int,
            min_sigma: float = 1.0,
            max_sigma: float = 1.0,
            decay_period: int = 1000000,
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)


class OUNoise:
    """Ornstein-Uhlenbeck process.
    referenced Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
            self,
            size: int,
            mu: float = 0.0,
            theta: float = 0.15,
            sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state
