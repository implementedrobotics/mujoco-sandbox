import mujoco
import mujoco_viewer

import math
import numpy as np

from transforms3d import euler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')


def get_sensor_offset(model: mujoco.MjModel, sensor_id: int):
    """
    Get the offset of the sensor in the sensor data array
    """
    offset = 0
    for i in range(sensor_id):
        offset += model.sensor_dim[i]
    return offset


class Encoder:
    """
    A class to simulate an incremental encoder for a Mujoco model.

    TODO: Add absolute encoder support

    Attributes:
        model (mujoco.MjModel): The Mujoco model.
        data (mujoco.MjData): The Mujoco data instance.
        sensor_id (int): The sensor ID for the joint to which the encoder is attached.
        ppr (int): Pulses per revolution for the encoder (default: 4096).
        index_angle (float): Angle in radians at which the index signal is triggered (default: 0).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, sensor_id: int, ppr: int = 4096, index_angle: float = 0):
        """Initialize an Encoder instance.

        Args:
            model (mujoco_py.MjModel): The Mujoco model.
            data (mujoco_py.MjData): The Mujoco data instance.
            sensor_id (int): The sensor ID for the joint to which the encoder is attached.
            ppr (int, optional): Pulses per revolution for the encoder. Defaults to 4096.
            index_angle (float, optional): Angle in radians at which the index signal is triggered. Defaults to 0.
        """

        self.model = model
        self.data = data
        self.sensor_id = sensor_id
        self.ppr = ppr
        self.index_count = int((index_angle / (2 * math.pi)) * self.ppr)
        self.index_value = 0
        self.sensor_offset = get_sensor_offset(model, sensor_id)
        self.encoder_value = 0

    def update(self):
        """Update the encoder's position and index signal values based on the joint position."""

        # Get the "real" joint position from our sensor
        joint_pos = self.data.sensordata[self.sensor_offset]

        # Convert to an encoder position
        self.encoder_value = int((joint_pos / (2 * math.pi)) * self.ppr)

        # Check for index signal
        self.index_value = 1 if self.encoder_value % self.ppr == self.index_count else 0

        print(f"{joint_pos} : {self.encoder_value} : {self.index_value}")

    def print(self):
        """Print the current encoder position."""
        print(f"Encoder: {self.encoder_value}")
        pass


# More legible printing from numpy
np.set_printoptions(precision=3, suppress=True, formatter={
                    'float': '{: 0.2f}'.format}, linewidth=300)

model = mujoco.MjModel.from_xml_path('encoder_test.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(
    model, data, width=1000, height=1000, hide_menus=True)

# Get the actuator and sensor IDs
rot_vel_id = model.actuator('rotational_velocity').id
sensor_id = model.sensor('rotary_encoder').id

# Create the encoder
encoder = Encoder(model, data, sensor_id=sensor_id,
                  ppr=1000, index_angle=0)

truth_values = []
sensor_values = []
encoder_values = []
ts = []

# simulate and render
for _ in range(1000):
    if viewer.is_alive:
        ts.append(data.time)

        # Set the rotational velocity
        data.ctrl[rot_vel_id] = 1.0

        # Step Simulation
        mujoco.mj_step(model, data)

        # Read/Update the encoder
        encoder.update()

        #  Store the values for plotting
        truth_values.append(data.qpos[0])
        sensor_values.append(data.sensordata[sensor_id])
        encoder_values.append(encoder.encoder_value /
                              encoder.ppr * (math.pi * 2))

        # Render
        viewer.render()

    else:
        break

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].step(ts, encoder_values, label='Encoder pos', where='post')
ax[0].plot(ts, truth_values, label='Truth')
ax[0].plot(ts, sensor_values, label='Sensor')

ax[0].set_ylabel('Encoder Estimation')
ax[0].legend()


plt.show()


# close
viewer.close()
