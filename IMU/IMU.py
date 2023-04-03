import mujoco
import mujoco_viewer

import math
import numpy as np

from transforms3d import euler


def get_sensor_offset(model, sensor_id):
    """
    Get the offset of the sensor in the sensor data array
    """
    offset = 0
    for i in range(sensor_id):
        offset += model.sensor_dim[i]
    return offset


# Class to hold the IMU data
# TODO: Add Sensor Fusion
class IMU:
    """
    Represents an Inertial Measurement Unit (IMU) with accelerometer, gyro, and magnetometer sensors.

    Attributes:
        model (mjModel): The Mujoco model object.
        data (mjData): The Mujoco data object.
        accel_id (int): The ID of the accelerometer sensor.
        gyro_id (int): The ID of the gyro sensor.
        mag_id (int): The ID of the magnetometer sensor.
        accel_offset (int): The offset of the accelerometer data in the sensordata array.
        gyro_offset (int): The offset of the gyro data in the sensordata array.
        mag_offset (int): The offset of the magnetometer data in the sensordata array.
        accel (np.ndarray): The current accelerometer data.
        gyro (np.ndarray): The current gyro data.
        mag (np.ndarray): The current magnetometer data.

    Methods:
        update(): Update the sensor data from the Mujoco simulation's sensordata array.
        print(): Print the current sensor data for accelerometer, gyro, and magnetometer.
    """

    def __init__(self, model, data, accel_id, gyro_id, mag_id):
        self.model = model
        self.data = data
        self.accel_id = accel_id
        self.gyro_id = gyro_id
        self.mag_id = mag_id

        self.accel_offset = get_sensor_offset(model, accel_id)
        self.gyro_offset = get_sensor_offset(model, gyro_id)
        self.mag_offset = get_sensor_offset(model, mag_id)

        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)

    def update(self):
        """
        Update the sensor data from the Mujoco simulation's sensordata array.
        """
        self.accel = self.data.sensordata[self.accel_offset:
                                          self.accel_offset + 3]
        self.gyro = self.data.sensordata[self.gyro_offset:
                                         self.gyro_offset + 3]
        self.mag = self.data.sensordata[self.mag_offset: self.mag_offset + 3]

    def compute_orientation(self):
        """
        Compute the orientation of the IMU using the magnetometer data.
        """

        # ax_ned = -self.accel[1]
        # ay_ned = self.accel[0]
        # az_ned = -self.accel[2]
        # g = np.array([ax_ned, ay_ned, -9.81])

        # Compute Pitch and Roll from Accelerometer
        pitch = math.asin(self.accel[0] / (-9.81))

        roll = math.atan(self.accel[1] / self.accel[2])

        # roll = math.atan2(-self.accel[0],
        #                   math.sqrt(self.accel[1]**2 + self.accel[2]**2))

        rot_matrix = data.xmat[1].reshape(3, 3)

        # pitch_check = np.arctan2(-rot_matrix[2, 0],
        #                          np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2))
        # roll_check = np.arctan2(
        #     rot_matrix[1, 0] / np.cos(pitch_check), rot_matrix[0, 0] / np.cos(pitch_check))
        # yaw_check = np.arctan2(rot_matrix[2, 1] / np.cos(pitch_check),
        #                        rot_matrix[2, 2] / np.cos(pitch_check))

        # print(pitch)
        # data.qpos[3] = roll
        # data.qpos[4] = pitch

        roll_check, pitch_check, yaw_check = euler.mat2euler(
            rot_matrix, axes='sxyz')

        # Quaternion representation
        qw = math.cos(roll/2) * math.cos(pitch/2)
        qx = math.sin(roll/2) * math.cos(pitch/2)
        qy = math.cos(roll/2) * math.sin(pitch/2)
        qz = math.sin(roll/2) * math.sin(pitch/2)

        body2_id = model.body('box_body_hat').id
        # data.mocap_quat[0] = np.array([qw, qx, qy, qz])

        data.mocap_quat[0] = euler.euler2quat(
            roll, pitch, 0, axes='sxyz')

        # print(f"Roll: {roll + math.pi}:{roll_check} and {pitch}:{pitch_check}")

        print(f"Roll: {math.degrees(roll)} | {math.degrees(roll_check)} -> Pitch: {math.degrees(pitch)} | {math.degrees(pitch_check)}")

    def print(self):
        """
        Print the current sensor data for accelerometer, gyro, and magnetometer.
        """
        print(f"Accel: {self.accel}")
        # print(f"Gyro : {self.gyro}")
        # print(f"Mag  : {self.mag}")

        self.compute_orientation()


# More legible printing from numpy
np.set_printoptions(precision=3, suppress=True, formatter={
                    'float': '{: 0.2f}'.format}, linewidth=300)

model = mujoco.MjModel.from_xml_path('IMU_test.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(
    model, data, width=1000, height=1000, hide_menus=True)

# Get the sensor idds
# accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'BMI088_ACC')
# gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'BMI088_GYR')
# mag_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'BMI088_MAG')
accel_id = model.sensor('BMI088_ACC').id
gyro_id = model.sensor('BMI088_GYR').id
mag_id = model.sensor('BMI088_MAG').id

# roll_vel_id = model.actuator('roll_velocity').id
# pitch_vel_id = model.actuator('pitch_velocity').id
# yaw_vel_id = model.actuator('yaw_velocity').id

# Accelorometer Noise
noise_density = 175e-6  # 175 µg/√Hz in g/√Hz (BMI088)
bandwidth = 100  # Hz [5 to 523Hz]
acc_noise = noise_density * math.sqrt(bandwidth)

# print(acc_noise)

# Compute Gyro Noise
noise_density = 0.014  # 0.014 dps/√Hz in rad/√Hz (BMI088)
bandwidth = 100  # Hz
gyro_noise = noise_density * math.sqrt(bandwidth)
gyro_noise = gyro_noise * math.pi / 180  # Convert to rad/s

# print(gyro_noise)

# model.sensor_noise[accel_id] = acc_noise
bmi088 = IMU(model, data, accel_id, gyro_id, mag_id)

data.qpos[:4] = euler.euler2quat(0, math.pi/2, 0, axes='sxyz')
# data.qpos[0] = math.radians(30)
# data.qpos[1] = math.radians(60)
# data.qpos[1] = math.pi/8
pitch_in = 0
# simulate and render
for _ in range(10000):
    if viewer.is_alive:

        # data.ctrl[roll_vel_id] = 2
        # data.ctrl[pitch_vel_id] = 2
        pitch_in = pitch_in + 0.01
        data.qpos[:4] = euler.euler2quat(0, pitch_in, 0, axes='sxyz')
        mujoco.mj_step(model, data)

        bmi088.update()
        bmi088.print()

        viewer.render()

    else:
        break

# close
viewer.close()
