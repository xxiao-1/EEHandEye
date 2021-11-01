#!/usr/bin/env python
import unittest

import numpy as np
import numpy.testing as npt

from hand_eye_calibration.dual_quaternion_hand_eye_calibration import (
    compute_hand_eye_calibration, compute_hand_eye_calibration_RANSAC, align_paths_at_index,
    compute_dual_quaternions_with_offset, HandEyeConfig)
from hand_eye_calibration.hand_eye_calibration_plotting_tools import (
    plot_alignment_errors, plot_poses)
from hand_eye_calibration.dual_quaternion import DualQuaternion
from hand_eye_calibration.quaternion import Quaternion
import hand_eye_calibration.hand_eye_test_helpers as he_helpers


class HandEyeCalibration(unittest.TestCase):
    # CONFIG
    paths_start_at_origin = True
    enforce_same_non_dual_scalar_sign = True
    enforce_positive_non_dual_scalar_sign = True
    make_plots_blocking = False

    # T1
    dq_H_E = he_helpers.random_transform_as_dual_quaternion(
        enforce_positive_non_dual_scalar_sign)
    assert dq_H_E.q_rot.w >= -1e-8

    pose_H_E = dq_H_E.to_pose()
    dq_H_E.normalize()

    # T2
    dq_H_E2 = he_helpers.random_transform_as_dual_quaternion(
        enforce_positive_non_dual_scalar_sign)
    assert dq_H_E2.q_rot.w >= -1e-8

    pose_H_E2 = dq_H_E.to_pose()
    dq_H_E2.normalize()

    dq_B_W = he_helpers.identity_transform_as_dual_quaternion()
    assert dq_B_W.q_rot.w >= -1e-8
    dq_B_H_vec, dq_W_E_vec = he_helpers.generate_test_paths(
        20, dq_H_E, dq_B_W, paths_start_at_origin)

    def test_hand_eye_calibration(self):
        # Generate A,B
        dq_B_H_vec1, dq_W_E_vec1 = he_helpers.generate_test_paths(
            300, self.dq_H_E, self.dq_B_W, self.paths_start_at_origin)

        # Generate C
        dq_B_H_vec_for_W_E1 = he_helpers.generate_test_path(
            300, False, 0.1, False, 0.1, 0.01)
        dq_W_E_vec2 = compute_dual_quaternions_with_offset(
            dq_B_H_vec_for_W_E1, self.dq_H_E2, self.dq_B_W)
        if self.paths_start_at_origin:
            dq_W_E_vec2 = align_paths_at_index(dq_W_E_vec2)

        # save path
        poses_H_B1 = np.array([dq_B_H_vec1[0].inverse().to_pose().T])
        poses_W_E1 = np.array([dq_W_E_vec1[0].to_pose().T])
        poses_W_E2 = np.array([dq_W_E_vec2[0].to_pose().T])

        for i in range(1, len(dq_B_H_vec1)):
            poses_H_B1 = np.append(poses_H_B1, np.array(
                [dq_B_H_vec1[i].inverse().to_pose().T]), axis=0)
            poses_W_E1 = np.append(poses_W_E1, np.array(
                [dq_W_E_vec1[i].inverse().to_pose().T]), axis=0)
            poses_W_E2 = np.append(poses_W_E2, np.array(
            [dq_W_E_vec2[i].inverse().to_pose().T]), axis=0)

        poses_H_B1 = np.insert(poses_H_B1, 0, values=np.arange(
            1, poses_H_B1.shape[0]+1), axis=1)
        poses_W_E1 = np.insert(poses_W_E1, 0, values=np.arange(
            1, poses_W_E1.shape[0]+1), axis=1)
        poses_W_E2 = np.insert(poses_W_E2, 0, values=np.arange(
            1, poses_W_E2.shape[0]+1), axis=1)
        np.savetxt('1.txt', poses_H_B1, delimiter=" ")
        np.savetxt('2.txt', poses_W_E1, delimiter=" ")
        np.savetxt('3.txt', poses_W_E1, delimiter=" ")
        print('saved-----------------------')
        print(poses_H_B1.shape)
        plot_poses([poses_H_B1[:, 1:], poses_W_E1[:, 1:],poses_W_E2[:, 1:]],blocking=self.make_plots_blocking)

        # print("dq_H_E ground truth: \n{}".format(self.dq_H_E))

        # hand_eye_config = HandEyeConfig()
        # hand_eye_config.visualize = False
        # hand_eye_config.ransac_max_number_iterations = 50
        # hand_eye_config.ransac_sample_size = 3
        # (success, dq_H_E_estimated, rmse,
        #  num_inliers, num_poses_kept,
        #  runtime, singular_values,
        #  bad_singular_values) = compute_hand_eye_calibration_RANSAC(
        #     dq_B_H_vec1, dq_W_E_vec1, hand_eye_config)
        # assert success, "Hand-eye calibration, failed!"

        # pose_H_E_estimated = dq_H_E_estimated.to_pose()
        # dq_H_E_estimated.normalize()

        # assert pose_H_E_estimated[6] > 0.0, (
        #     "The real part of the pose's quaternion should be positive. "
        #     "The pose is: \n{}\n where the dual quaternion was: "
        #     "\n{}".format(pose_H_E_estimated, dq_H_E_estimated))

        # print("The static input pose was: \n{}".format(self.pose_H_E))
        # print("The hand-eye calibration's output pose is: \n{}".format(
        #       pose_H_E_estimated))

        # print("T_H_E ground truth: \n{}".format(self.dq_H_E.to_matrix()))
        # print("T_H_E estimated: \n{}".format(dq_H_E_estimated.to_matrix()))

        # assert np.allclose(
        #     self.dq_H_E.dq, dq_H_E_estimated.dq, atol=1e-3), (
        #     "input dual quaternion: {}, estimated dual quaternion: {}".format(
        #         self.dq_H_E, dq_H_E_estimated))

    # def test_hand_eye_calibration_with_outliers(self):
    #     dq_B_H_vec, dq_W_E_vec = he_helpers.generate_test_paths(
    #         20, self.dq_H_E, self.dq_B_W, self.paths_start_at_origin,
    #         include_outliers_B_H=True, outlier_probability_B_H=0.1,
    #         include_outliers_W_E=True, outlier_probability_W_E=0.2)

    #     # Plot the poses with outliers.
    #     poses_B_H = np.array([dq_B_H_vec[0].to_pose().T])
    #     poses_W_E = np.array([dq_W_E_vec[0].to_pose().T])
    #     for i in range(1, len(dq_B_H_vec)):
    #         poses_B_H = np.append(poses_B_H, np.array(
    #             [dq_B_H_vec[i].to_pose().T]), axis=0)
    #         poses_W_E = np.append(poses_W_E, np.array(
    #             [dq_W_E_vec[i].to_pose().T]), axis=0)
    #     # plot_poses([poses_B_H, poses_W_E], blocking=self.make_plots_blocking)

    #     hand_eye_config = HandEyeConfig()
    #     hand_eye_config.visualize = False
    #     hand_eye_config.ransac_max_number_iterations = 50
    #     hand_eye_config.ransac_sample_size = 3
    #     (success, dq_H_E_estimated, rmse,
    #      num_inliers, num_poses_kept,
    #      runtime, singular_values,
    #      bad_singular_values) = compute_hand_eye_calibration_RANSAC(dq_B_H_vec, dq_W_E_vec,
    #                                                                 hand_eye_config)
    #     assert success, "Hand-eye calibration, failed!"

    #     pose_H_E_estimated = dq_H_E_estimated.to_pose()
    #     dq_H_E_estimated.normalize()

    #     assert pose_H_E_estimated[6] > 0.0, (
    #         "The real part of the pose's quaternion should be positive. "
    #         "The pose is: \n{}\n where the dual quaternion was: "
    #         "\n{}".format(pose_H_E_estimated, dq_H_E_estimated))

    #     print("The static input pose was: \n{}".format(self.pose_H_E))
    #     print("The hand-eye calibration's output pose is: \n{}".format(
    #           pose_H_E_estimated))

    #     print("T_H_E ground truth: \n{}".format(self.dq_H_E.to_matrix()))
    #     print("T_H_E estimated: \n{}".format(dq_H_E_estimated.to_matrix()))

    #     assert np.allclose(
    #         self.dq_H_E.dq, dq_H_E_estimated.dq, atol=2e-2), (
    #         "input dual quaternion: {}, estimated dual quaternion: {}".format(
    #             self.dq_H_E, dq_H_E_estimated))

    # def test_hand_eye_calibration_with_noise(self):
    #     dq_B_H_vec, dq_W_E_vec = he_helpers.generate_test_paths(
    #         20, self.dq_H_E, self.dq_B_W, self.paths_start_at_origin,
    #         include_outliers_B_H=False, include_noise_B_H=True,
    #         noise_sigma_trans_B_H=0.0001, noise_sigma_rot_B_H=0.001,
    #         include_outliers_W_E=False, include_noise_W_E=True,
    #         noise_sigma_trans_W_E=0.0001, noise_sigma_rot_W_E=0.001)

    #     # Plot the poses with noise.
    #     poses_B_H = np.array([dq_B_H_vec[0].to_pose().T])
    #     poses_W_E = np.array([dq_W_E_vec[0].to_pose().T])
    #     for i in range(1, len(dq_B_H_vec)):
    #         poses_B_H = np.append(poses_B_H, np.array(
    #             [dq_B_H_vec[i].to_pose().T]), axis=0)
    #         poses_W_E = np.append(poses_W_E, np.array(
    #             [dq_W_E_vec[i].to_pose().T]), axis=0)
    #     # plot_poses([poses_B_H, poses_W_E], blocking=self.make_plots_blocking)

    #     hand_eye_config = HandEyeConfig()
    #     hand_eye_config.visualize = False
    #     hand_eye_config.ransac_max_number_iterations = 50
    #     hand_eye_config.ransac_sample_size = 3
    #     (success, dq_H_E_estimated, rmse,
    #      num_inliers, num_poses_kept,
    #      runtime, singular_values,
    #      bad_singular_values) = compute_hand_eye_calibration_RANSAC(dq_B_H_vec, dq_W_E_vec,
    #                                                                 hand_eye_config)
    #     assert success, "Hand-eye calibration, failed!"

    #     pose_H_E_estimated = dq_H_E_estimated.to_pose()
    #     dq_H_E_estimated.normalize()

    #     assert pose_H_E_estimated[6] > 0.0, (
    #         "The real part of the pose's quaternion should be positive. "
    #         "The pose is: \n{}\n where the dual quaternion was: "
    #         "\n{}".format(pose_H_E_estimated, dq_H_E_estimated))

    #     print("The static input pose was: \n{}".format(self.pose_H_E))
    #     print("The hand-eye calibration's output pose is: \n{}".format(
    #           pose_H_E_estimated))

    #     print("T_H_E ground truth: \n{}".format(self.dq_H_E.to_matrix()))
    #     print("T_H_E estimated: \n{}".format(dq_H_E_estimated.to_matrix()))

    #     assert np.allclose(
    #         self.dq_H_E.dq, dq_H_E_estimated.dq, atol=4e-2), (
    #         "input dual quaternion: {}, estimated dual quaternion: {}".format(
    #             self.dq_H_E, dq_H_E_estimated))

    #     # dq_W_E_vec = compute_dual_quaternions_with_offset(
    #     #     dq_B_H_vec, dq_H_E_estimated, self.dq_B_W)
    #     # # print(dq_W_E_vec)
    #     # pose_W_E_estimated = np.array([dq_W_E_vec[0].to_pose().T])
    #     # for i in range(1, len(dq_W_E_vec)):
    #     #   pose_W_E_estimated = np.append(poses_B_H, np.array(
    #     #       [dq_W_E_vec[i].to_pose().T]), axis=0)
    #     # plot_poses([pose_W_E_estimated,poses_B_H], blocking=self.make_plots_blocking)
    #     # plot_poses([pose_W_E_estimated], blocking=self.make_plots_blocking)

    # def test_plot_poses(self):
    #     # Draw both paths in their Global/World frame.
    #     poses_B_H = np.array([self.dq_B_H_vec[0].to_pose().T])
    #     poses_W_E = np.array([self.dq_W_E_vec[0].to_pose().T])
    #     for i in range(1, len(self.dq_B_H_vec)):
    #         poses_B_H = np.append(poses_B_H, np.array(
    #             [self.dq_B_H_vec[i].to_pose().T]), axis=0)
    #         poses_W_E = np.append(poses_W_E, np.array(
    #             [self.dq_W_E_vec[i].to_pose().T]), axis=0)
    #     # plot_poses([poses_B_H, poses_W_E], blocking=self.make_plots_blocking)

    # def test_save_pose(self):
    #     poses_H_B = np.array([self.dq_B_H_vec[0].inverse().to_pose().T])
    #     poses_W_E = np.array([self.dq_W_E_vec[0].to_pose().T])

    #     for i in range(1, len(self.dq_B_H_vec)):
    #         poses_H_B=np.append(poses_H_B, np.array(
    #             [self.dq_B_H_vec[i].inverse().to_pose().T]), axis=0)
    #         poses_W_E=np.append(poses_W_E, np.array(
    #             [self.dq_W_E_vec[i].inverse().to_pose().T]), axis=0)

    #     np.savetxt('1.txt', poses_H_B, delimiter=" ")
    #     print('saved-----------------------')


if __name__ == '__main__':
    unittest.main()
