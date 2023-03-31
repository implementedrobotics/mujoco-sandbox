import mujoco
import mujoco_viewer

import math
import numpy as np


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

    def print(self):
        """
        Print the current sensor data for accelerometer, gyro, and magnetometer.
        """
        print(f"Accel: {self.accel}")
        print(f"Gyro : {self.gyro}")
        print(f"Mag  : {self.mag}")


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

roll_vel_id = model.actuator('roll_velocity').id
pitch_vel_id = model.actuator('pitch_velocity').id
yaw_vel_id = model.actuator('yaw_velocity').id

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

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        data.ctrl[roll_vel_id] = 1
        mujoco.mj_step(model, data)
        viewer.render()
        bmi088.update()
        bmi088.print()

    else:
        break

# close
viewer.close()
