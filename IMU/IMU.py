import mujoco
import mujoco_viewer

import math
import numpy as np

from transforms3d import euler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')


def get_sensor_offset(model, sensor_id):
    """
    Get the offset of the sensor in the sensor data array
    """
    offset = 0
    for i in range(sensor_id):
        offset += model.sensor_dim[i]
    return offset


# Class to hold the IMU data
class IMU:
    """
    Represents an Inertial Measurement Unit (IMU) with accelerometer, gyro, and magnetometer sensors integrated with a Mujoco simulation.

    Attributes:
        model (mjModel): The Mujoco model object.
        data (mjData): The Mujoco data object.
        accel_id (int): The ID of the accelerometer sensor.
        gyro_id (int): The ID of the gyro sensor.
        mag_id (int): The ID of the magnetometer sensor.
        gyro_bias (np.ndarray): The constant bias of the gyro sensor.
        dt (float): The sample time interval for updating sensor data.
        alpha (float, optional): The complementary filter's blending factor. Defaults to 0.5.
        q0 (np.ndarray, optional): The initial quaternion for orientation. Defaults to None.
        accel_offset (int): The offset of the accelerometer data in the sensordata array.
        gyro_offset (int): The offset of the gyro data in the sensordata array.
        mag_offset (int): The offset of the magnetometer data in the sensordata array.
        accel (np.ndarray): The current accelerometer data.
        gyro (np.ndarray): The current gyro data.
        mag (np.ndarray): The current magnetometer data.

    Methods:
        update(): Update the sensor data from the Mujoco simulation's sensordata array and compute the current orientation.
        get_euler(): Convert the quaternion orientation to Euler angles in the RZYX order (yaw, pitch, roll).
        print(): Print the current sensor data for accelerometer, gyro, and magnetometer (currently not used).
    """

    def __init__(self, model, data, accel_id, gyro_id, mag_id, gyro_bias, dt, alpha=0.5, q0=None):
        self.model = model
        self.data = data
        self.accel_id = accel_id
        self.gyro_id = gyro_id
        self.mag_id = mag_id

        if q0 is None:
            self.q = euler.euler2quat(0, 0, 0, axes='rzyx')
        else:
            self.q = q0

        self.dt = dt
        self.alpha = alpha
        self.gyro_bias = gyro_bias
        self.next_sample_time = self.dt

        self.accel_offset = get_sensor_offset(model, accel_id)
        self.gyro_offset = get_sensor_offset(model, gyro_id)

        if self.mag_id is not None:
            self.mag_offset = get_sensor_offset(model, mag_id)

        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)

    def update(self):
        """
        Update the sensor data from the Mujoco simulation's sensordata array.
        """

        # Compute data depending on sample rate
        if data.time >= self.next_sample_time:
            self.next_sample_time += self.dt

            self.accel = self.data.sensordata[self.accel_offset:
                                              self.accel_offset + 3]
            self.gyro = self.data.sensordata[self.gyro_offset:
                                             self.gyro_offset + 3]

            if self.mag_id is not None:
                self.mag = self.data.sensordata[self.mag_offset:
                                                self.mag_offset + 3]

            self.compute_orientation()

    def compute_orientation(self):
        """
        Compute the orientation of the IMU using a complimentary filter with the accelerometer and gyro data and optional magnetometer data.
        """

        # Compute Pitch and Roll from Gyro with attitude propagation for quaternion

        # Cache gyro data
        w_x = self.gyro[0] + self.gyro_bias[0]
        w_y = self.gyro[1] + self.gyro_bias[1]
        w_z = self.gyro[2] + self.gyro_bias[2]

        # Cache current quaternion data
        q_w = self.q[0]
        q_x = self.q[1]
        q_y = self.q[2]
        q_z = self.q[3]

        # Compute attitude propagation
        q_omega_w = q_w - 0.5 * self.dt * w_x * q_x - 0.5 * \
            self.dt * w_y * q_y - 0.5 * self.dt * w_z * q_z
        q_omega_x = q_x + 0.5 * self.dt * w_x * q_w - 0.5 * \
            self.dt * w_y * q_z + 0.5 * self.dt * w_z * q_y
        q_omega_y = q_y + 0.5 * self.dt * w_x * q_z + 0.5 * \
            self.dt * w_y * q_w - 0.5 * self.dt * w_z * q_x
        q_omega_z = q_z - 0.5 * self.dt * w_x * q_y + 0.5 * \
            self.dt * w_y * q_x + 0.5 * self.dt * w_z * q_w

        # Quaternion Estimate from Gyro
        q_omega = np.array([q_omega_w, q_omega_x, q_omega_y, q_omega_z])

        # Compute Pitch and Roll from Accelerometer
        ax = self.accel[0]
        ay = self.accel[1]
        az = self.accel[2]

        # Yaw
        psi = 0.0

        # Pitch
        theta = math.atan2(-ax, math.sqrt(ay**2 + az**2))

        # Roll
        phi = math.atan2(ay, az)

        # Correct Yaw from Magnetometer (if available)
        if self.mag_id is not None:
            b_x = self.mag[0] * math.cos(theta) + self.mag[1] * math.sin(
                theta) * math.sin(phi) + self.mag[2] * math.sin(theta) * math.cos(phi)
            b_y = self.mag[1] * math.cos(phi) - self.mag[2] * math.sin(phi)

            psi = math.atan2(-b_y, b_x)

        # Quaternion Estimate from Accelerometer
        q_am = euler.euler2quat(psi, theta, phi, axes='rzyx')

        self.q = q_omega * (1 - self.alpha) + q_am * self.alpha

    def get_euler(self):
        """
        Convert the quaternion orientation to Euler angles in the RZYX order (yaw, pitch, roll).

        Returns:
            tuple: A tuple of three float values representing the Euler angles (yaw, pitch, roll) in radians.
        """

        return euler.quat2euler(self.q, axes='rzyx')

    def print(self):
        """
        Print the current sensor data for accelerometer, gyro, and magnetometer.
        """
        # print(f"Accel: {self.accel}")
        # print(f"Gyro : {self.gyro}")
        # print(f"Mag  : {self.mag}")
        pass


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

# roll_motor_id = model.actuator('roll_motor').id
# pitch_motor_id = model.actuator('pitch_motor').id
# yaw_motor_id = model.actuator('yaw_motor').id

# Accelorometer Noise
noise_density = 175e-6  # 175 µg/√Hz in g/√Hz (BMI088)
# noise_density = 0.14e-3  # 0.14 mg/√Hz in g/√Hz (VN100)
bandwidth = 100  # Hz [5 to 523Hz]
acc_noise = noise_density * math.sqrt(bandwidth)

# Compute Gyro Noise
noise_density = 0.014  # 0.014 dps/√Hz in rad/√Hz (BMI088)
# noise_density = 0.0035  # 0.014 dps/√Hz in rad/√Hz (VN100)
bandwidth = 100  # Hz
gyro_noise = noise_density * math.sqrt(bandwidth)
gyro_noise = gyro_noise * math.pi / 180  # Convert to rad/s

# Magnetometer Noise
noise_density = 0.6e-6  # 0.6 µT/√Hz in T/√Hz (BMI088)
# noise_density = 140e-6 / 10000  # 140 µT/√Hz in T/√Hz (VN100)
bandwidth = 100  # Hz
mag_noise = noise_density * math.sqrt(bandwidth)

# model.sensor_noise[accel_id] = acc_noise

gyro_bias = 1.0 * math.pi / 180.0  # dps -> rps

q0 = euler.euler2quat(0, 0, 0, axes='rzyx')
bmi088 = IMU(model, data, accel_id, gyro_id, mag_id, gyro_bias=[gyro_bias, gyro_bias, gyro_bias],
             dt=0.01, alpha=0.05, q0=q0)


ts = []
theta_data = []
phi_data = []
psi_data = []
truth_psi_data = []
truth_theta_data = []
truth_phi_data = []


# data.qpos[:4] = q0

# simulate and render
for _ in range(1000):
    if viewer.is_alive:
        ts.append(data.time)

        # data.ctrl[roll_motor_id] = 1
        # data.ctrl[pitch_motor_id] = 1
        # data.qpos[:4] = q0
        mujoco.mj_step(model, data)

        bmi088.update()

        # Get roll, pitch, yaw
        psi_data.append(math.degrees(bmi088.get_euler()[0]))
        theta_data.append(math.degrees(bmi088.get_euler()[1]))
        phi_data.append(math.degrees(bmi088.get_euler()[2]))

        # Get ground truth roll, pitch, yaw
        truth_euler = euler.quat2euler(data.qpos[0:4], axes='rzyx')
        truth_psi_data.append(math.degrees(truth_euler[0]))
        truth_theta_data.append(math.degrees(truth_euler[1]))
        truth_phi_data.append(math.degrees(truth_euler[2]))

        viewer.render()

    else:
        break

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(ts, psi_data, label='Yaw')
ax[0].plot(ts, theta_data, label='Pitch')
ax[0].plot(ts, phi_data, label='Roll')

ax[0].plot(ts, truth_psi_data, label='Gnd Yaw')
ax[0].plot(ts, truth_theta_data, label='Gnd Pitch')
ax[0].plot(ts, truth_phi_data, label='Gnd Roll')

ax[0].set_ylabel('Euler Angles (rad)')
ax[0].legend()


plt.show()


# close
viewer.close()
