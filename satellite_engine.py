import math
from os.path import join as pjoin

import numpy as np
import scipy.io as sio
import wandb
from scipy.integrate import odeint

import satellite_constant
from train_utils import Quaternions2EulerAngles, EulerAngles2Quaternions, Quaternions2RotationMatrix
from train_utils import RotationMatrix2SixD


class Satellite:
    """This is simplified version of the satellite model"""

    def __init__(self, inc_angle, num_of_planes, plane_start, sat_per_plane, time_steps_per_action, altitude,
                 number_episodes):
        self.inc_angle = inc_angle
        self.num_of_planes = num_of_planes
        self.plane_start = plane_start
        self.sat_per_plane = sat_per_plane
        self.time_steps_per_action = time_steps_per_action
        self.Constant = satellite_constant.Constant()
        self.altitude = altitude
        self.Idot = 0
        self.state = {}
        self.number_episodes = number_episodes
        self.Inertia = np.array([[1.12, 0.15, -1.1], [0.12, 10.59, -5.15], [-10.1, -5.15, 15.96]])
        self.Inertia_inv = np.linalg.inv(self.Inertia)
        self.sat_m = 27.7
        self.F_solar = 0
        self.num_coil = 2
        self.magnetorque_area = 1
        mat_fname = pjoin('mag_field.mat')
        mat_contents = sio.loadmat(mat_fname)
        self.BX = np.array(mat_contents['Bx'])
        self.BY = np.array(mat_contents['By'])
        self.BZ = np.array(mat_contents['Bz'])

    def main_eqs(self, state_solver, time, actions, Bx, By, Bz):

        # Quatornian modeling
        Bxyz = (np.array([Bx, By, Bz])) / 1e9

        # Magnotorque modeling
        mu_xy = 16.8
        mu_z = 18.4

        ix = actions[0] * mu_xy * 1  # This is Am^2
        iy = actions[1] * mu_xy * 1  # This is Am^2
        iz = actions[2] * mu_z * 1  # This is Am^2

        current = np.array([ix, iy, iz])
        M_m = np.cross(Bxyz, current)
        H = np.matmul(self.Inertia, omega)
        quat_der = [[0, -omega[0], -omega[1], -omega[2]], [omega[0], 0, omega[2], -omega[1]],
                    [omega[1], -omega[2], 0, omega[0]], [omega[2], omega[1], -omega[0], 0]]
        Mom_tot = M_m - np.cross(omega, H)

        omega_dot = np.matmul(np.linalg.inv(self.Inertia), Mom_tot)

        return np.concatenate((quat_dot, omega_dot), axis=0)

    def J2_perturbation(self, r_vec):
        z2 = np.square(r_vec[2])
        norm_r = np.linalg.norm(r_vec)
        r2 = np.square(norm_r)
        tx = r_vec[0] / norm_r * (5 * z2 / r2 - 1)
        ty = r_vec[1] / norm_r * (5 * z2 / r2 - 1)
        tz = r_vec[2] / norm_r * (5 * z2 / r2 - 3)
        acc_J2 = -1.5 * self.Constant.J2 * self.Constant.gravity * self.Constant.earth_mass * np.square(
            self.Constant.earthRadius) / np.power(norm_r, 4) * np.array([tx, ty, tz])
        return acc_J2

    def Cartesian_latlongalt(self, orbital_pos, time):
        """
        Transfer cartesian coordinate to lattitude longitude and altitude considering the earth rotation
        """
        orb_x = orbital_pos[:, 0]
        orb_y = orbital_pos[:, 1]
        orb_z = orbital_pos[:, 2]
        E_Per = 24 * 3600
        lon_shift = (360 * time / E_Per) % 360
        rho = np.linalg.norm(orbital_pos, axis=1)
        thetaE = np.arccos(orb_z / rho)
        saiE = np.arctan2(orb_y, orb_x)
        lat = 90 - thetaE * 180 / np.pi
        lon = saiE * 180 / np.pi
        alt = rho - self.Constant.earthRadius
        lon -= lon_shift
        lon += 180
        lon = lon % 360
        lon -= 180
        return lat, lon, alt

    def main_solver(self, state_solver, action, time, Bx, By, Bz):
        next_state = odeint(self.main_eqs, state_solver, time, args=(action, Bx, By, Bz))
        return next_state[-1, :]

    def get_new_state(self, state_solver, actions, time_record, lat, lon, B_xyz, control_trigger):
        desired_quat = np.array([1, 0, 0, 0])
        desired_euler = np.array([0, 0, 0])
        desired_omega = np.array([0, 0, 0])
        lat_ind = (lat[time_record + 1]) / 90
        lon_ind = (lon[time_record + 1]) / 180
        prev_quat = state_solver[0:4]
        prev_omega = state_solver[4:7]
        prev_euler = Quaternions2EulerAngles(prev_quat)

        Bx = B_xyz[time_record + 1, 0]
        By = B_xyz[time_record + 1, 1]
        Bz = B_xyz[time_record + 1, 2]

        # controller actions
        inv_inertia = self.Inertia_inv
        current1 = -np.cross(np.array([prev_omega[0], prev_omega[1], prev_omega[2]]),
                             np.array([Bx, By, Bz])) * 1e6 / 1e9
        quat_vec = np.array(
            [inv_inertia[0, 0] * prev_quat[1] / prev_quat[0], inv_inertia[1, 1] * prev_quat[2] / prev_quat[0],
             inv_inertia[2, 2] * prev_quat[3] / prev_quat[0]])
        current2 = -np.cross(quat_vec, np.array([Bx, By, Bz])) * 1e5 / 1e9
        final_current = current1 + current2
        controller_actions = np.clip(final_current, -1, 1)
        wandb.log({'controller_actions_x': controller_actions[0]})
        wandb.log({'controller_actions_y': controller_actions[1]})
        wandb.log({'controller_actions_z': controller_actions[2]})

        omega_trigger_1 = 0.01
        omega_trigger_2 = 0.005
        euler_trigger_1 = 0.3
        euler_trigger_2 = 0.1
        if not control_trigger:
            if abs(prev_omega[0]) > omega_trigger_1 or abs(prev_omega[1]) > omega_trigger_1 or abs(
                    prev_omega[2]) > omega_trigger_1 \
                    or abs(prev_euler[0]) > euler_trigger_1 or abs(prev_euler[1]) > euler_trigger_1 or abs(
                prev_euler[2]) > euler_trigger_1:
                actions = controller_actions
                control_trigger = 1
        else:
            if abs(prev_omega[0]) > omega_trigger_2 or abs(prev_omega[1]) > omega_trigger_2 or abs(
                    prev_omega[2]) > omega_trigger_2 \
                    or abs(prev_euler[0]) > euler_trigger_2 or abs(prev_euler[1]) > euler_trigger_2 or abs(
                prev_euler[2]) > euler_trigger_2:
                actions = controller_actions
            else:
                control_trigger = 0

        time = np.linspace(0, self.time_steps_per_action, 2)
        next_state_solver = self.main_solver(state_solver, actions, time, Bx, By, Bz)

        actual_quat = np.array([next_state_solver[0], next_state_solver[1], next_state_solver[2], next_state_solver[3]])
        actual_euler = Quaternions2EulerAngles(actual_quat)
        actual_omega = np.array([next_state_solver[4], next_state_solver[5], next_state_solver[6]])
        rot_mat = Quaternions2RotationMatrix(actual_quat)
        rot_6d = RotationMatrix2SixD(rot_mat).flatten()
        error_euler = actual_euler - desired_euler
        error_omega = actual_omega - desired_omega

        magnetic_norm = 5e4
        omega_shift = 0.0

        next_state_RL = np.array(
            [Bx / magnetic_norm, By / magnetic_norm, Bz / magnetic_norm,
             rot_6d[0], rot_6d[1], rot_6d[2], rot_6d[3], rot_6d[4], rot_6d[5],
             error_omega[0] + omega_shift, error_omega[1] + omega_shift, error_omega[2] + omega_shift])

        reward = 0

        # Termination
        if abs(actual_omega[0]) > 0.5 or abs(actual_omega[1]) > 0.5 or abs(actual_omega[2]) > 0.5:
            status = True
        else:
            status = False

        # Bonus
        if abs(error_euler[0]) < 0.1 and abs(error_euler[1]) < 0.1 and abs(error_euler[2]) < 0.1:
            reward += 1

        # Continuous Reward
        reward += np.exp(-np.absolute(error_euler[0]) ** 2 / (2 * 1.3 ** 2)) * \
                  np.exp(-np.absolute(error_euler[1]) ** 2 / (2 * 1.3 ** 2)) * \
                  np.exp(-np.absolute(error_euler[2]) ** 2 / (2 * 1.3 ** 2))

        # Penalty
        reward -= 0.1 * np.linalg.norm(actions) / np.linalg.norm(error_euler)

        wandb.log({'action_x': actions[0]})
        wandb.log({'action_y': actions[1]})
        wandb.log({'action_z': actions[2]})
        wandb.log({'B_x': Bx})
        wandb.log({'B_y': By})
        wandb.log({'B_z': Bz})

        return next_state_RL, reward, next_state_solver, status, time_record + 1, controller_actions, control_trigger

    def Orbit_propagation(self):
        mu_earth = self.Constant.gravity * self.Constant.earth_mass
        semi_major_earth = 650e3 + self.Constant.earthRadius
        sat_period = 2 * np.pi * np.power(semi_major_earth, 1.5) / np.sqrt(mu_earth)

        final_time = self.number_episodes * self.time_steps_per_action
        time = np.linspace(0, final_time, self.number_episodes + 1)

        orbits = np.zeros((self.number_episodes + 1, 3))
        orbits[:, 0] = -semi_major_earth * np.cos(time / sat_period * 2 * math.pi)
        orbits[:, 2] = semi_major_earth * np.sin(time / sat_period * 2 * math.pi)

        rotationX = np.array([[1, 0, 0], [0, np.cos(self.inc_angle - 90), np.sin(self.inc_angle - 90)],
                              [0, -np.sin(self.inc_angle - 90), np.cos(self.inc_angle - 90)]])
        orbits = np.matmul(orbits, rotationX)

        lat, lon, alt = self.Cartesian_latlongalt(orbits, time)

        B_xyz = np.zeros((self.number_episodes + 1, 3))
        for index in range(len(time)):
            lat_rounded = np.round(lat[index])
            lon_rounded = np.round(lon[index])
            lat_index = lat_rounded.astype(int)
            lon_index = lon_rounded.astype(int)
            B_xyz[index, 0] = self.BX[180 + lon_index, 90 + lat_index]
            B_xyz[index, 1] = self.BY[180 + lon_index, 90 + lat_index]
            B_xyz[index, 2] = self.BZ[180 + lon_index, 90 + lat_index]

        return orbits, time, lat, lon, alt, B_xyz

    def initialize_sat(self, lat, lon, alt, B_xyz):

        phi0 = (2 * np.random.rand() - 1) * np.pi / 10
        theta0 = (2 * np.random.rand() - 1) * np.pi / 10
        sai0 = (2 * np.random.rand() - 1) * np.pi / 10
        w_p0 = (2 * np.random.rand() - 1) * 0.2
        w_y0 = (2 * np.random.rand() - 1) * 0.2
        w_r0 = (2 * np.random.rand() - 1) * 0.2

        pts0 = np.array([phi0, theta0, sai0])
        q0 = EulerAngles2Quaternions(np.transpose(pts0))
        w_vec = np.array([w_p0, w_y0, w_r0])
        lat_lon_0 = np.array([(lat[0]) / 90, (lon[0]) / 180, 0.65])
        B_xyz0 = np.array([B_xyz[0, 0], B_xyz[0, 1], B_xyz[0, 2]])
        rot_mat = Quaternions2RotationMatrix(q0)
        rot_6d = RotationMatrix2SixD(rot_mat).flatten()
        state_solver = np.concatenate((q0, w_vec), axis=0)

        # controller actions
        inv_inertia = self.Inertia_inv
        current1 = -np.cross(w_vec, B_xyz0) * 1e6 / 1e9
        quat_vec = np.array(
            [inv_inertia[0, 0] * q0[1] / q0[0], inv_inertia[1, 1] * q0[2] / q0[0],
             inv_inertia[2, 2] * q0[3] / q0[0]])
        current2 = -np.cross(quat_vec, B_xyz0) * 1e5 / 1e9
        final_current = current1 + current2
        controller_actions = np.clip(final_current, -1, 1)

        magnetic_norm = 5e4
        omega_shift = 0.0
        B_xyz0 = B_xyz0 / magnetic_norm
        w_vec = w_vec + omega_shift
        state_RL = np.concatenate((B_xyz0, rot_6d, w_vec), axis=0)

        action_mag = 5000
        action_set = np.array(
            [[0, 0, 0], [action_mag, 0, 0], [-action_mag, 0, 0], [0, action_mag, 0], [0, -action_mag, 0],
             [0, 0, action_mag], [0, 0, -action_mag]])
        time_record = 0

        return state_solver, state_RL, controller_actions, action_set, time_record
