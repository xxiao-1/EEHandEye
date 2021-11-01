from hand_eye_calibration.quaternion import Quaternion
import numpy as np
import csv


def read_time_stamped_poses_from_csv_file(csv_file, JPL_quaternion_format=False):
    """
    Reads time stamped poses from a CSV file.
    Assumes the following line format:
      timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw
    The quaternion is expected in Hamilton format, if JPL_quaternion_format is True
    it expects JPL quaternions and they will be converted to Hamiltonian quaternions.
    """
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        time_stamped_poses = np.array(list(csv_reader))
        time_stamped_poses = time_stamped_poses.astype(float)
    # print(time_stamped_poses)
    # Extract the quaternions from the poses.
    times = time_stamped_poses[:, 0].copy()
    poses = time_stamped_poses[:, 1:]

    quaternions = []
    for pose in poses:
        pose[3:] /= np.linalg.norm(pose[3:])
        if JPL_quaternion_format:
            quaternion_JPL = np.array([-pose[3], -pose[4], -pose[5], pose[6]])
            quaternions.append(Quaternion(q=quaternion_JPL))
        else:
            quaternions.append(Quaternion(q=pose[3:]))

    return (time_stamped_poses.copy(), times, quaternions)


def read_time_stamped_angular_velocity_from_csv_file(csv_file):
    """
    Reads time stamped poses from a CSV file.
    Assumes the following line format:
      timestamp [s], r, p, y
    """
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        time_stamped_poses = np.array(list(csv_reader))
        time_stamped_poses = time_stamped_poses.astype(float)

    # print(time_stamped_poses)
    # Extract the quaternions from the poses.
    times = time_stamped_poses[:, 0].copy()
    angV = time_stamped_poses[:, 1:].copy()

    return (times, angV)


def read_time_stamped_poses_from_txt_file(csv_file, JPL_quaternion_format=False):
    """
    Reads time stamped poses from a txt file.
    Assumes the following line format:
      timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw
    The quaternion is expected in Hamilton format, if JPL_quaternion_format is True
    it expects JPL quaternions and they will be converted to Hamiltonian quaternions.
    """

    time_stamped_poses = []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(' ')
            v=[float(line[0]),float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7])]
            time_stamped_poses.append(v)

    # Extract the quaternions from the poses.
    # print(time_stamped_poses)
    time_stamped_poses = np.array(time_stamped_poses)
    times = time_stamped_poses[:, 0].copy()
    poses = time_stamped_poses[:, 1:]

    quaternions = []
    for pose in poses:
        pose[3:] /= np.linalg.norm(pose[3:])
        if JPL_quaternion_format:
            quaternion_JPL = np.array([-pose[3], -pose[4], -pose[5], pose[6]])
            quaternions.append(Quaternion(q=quaternion_JPL))
        else:
            quaternions.append(Quaternion(q=pose[3:]))

    return (time_stamped_poses.copy(), times, quaternions)

def read_time_stamped_angular_velocity_from_txt_file(csv_file):
    """
    Reads time stamped poses from a txt file.
    Assumes the following line format:
      timestamp [s], r, p, y
    """
    time_stamped_poses = []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(' ')
            v=[float(line[0]),float(line[1]),float(line[2]),float(line[3])]
            time_stamped_poses.append(v)

    # Extract the quaternions from the poses.
    time_stamped_poses = np.array(time_stamped_poses)
    times = time_stamped_poses[:, 0].copy()
    angV = time_stamped_poses[:, 1:].copy()

    return (times, angV)

def write_double_numpy_array_to_csv_file(array, csv_file):
    np.savetxt(csv_file, array, delimiter=", ", fmt="%.18f")


def write_time_stamped_poses_to_csv_file(time_stamped_poses, csv_file):
    """
    Writes time stamped poses to a CSV file.
    Uses the following line format:
      timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw
    """
    write_double_numpy_array_to_csv_file(time_stamped_poses, csv_file)
