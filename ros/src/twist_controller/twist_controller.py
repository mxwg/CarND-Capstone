from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        wheel_base 	     = kwargs['wheel_base']
        self.steer_ratio = kwargs['steer_ratio']
        min_velocity     = kwargs['min_velocity']
        max_lat_accel    = kwargs['max_lat_accel']
        max_steer_angle  = kwargs['max_steer_angle']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        self.brake_deadband = kwargs['brake_deadband']
        self.vehicle_mass = kwargs['vehicle_mass']
        self.wheel_radius = kwargs['wheel_radius']

        self.yaw_controller = YawController(wheel_base, self.steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.vel_pid = PID(0.3, 0.1, 0., 0.0, self.accel_limit)
        self.lowpassFilt = LowPassFilter(0.5, 0.02)

    def control(self, proposed_linear_vel, proposed_angular_vel, current_linear_vel):
        # Acceleration Controller
        brake = 0.
        delta_throttle = self.vel_pid.step(proposed_linear_vel-current_linear_vel, 0.02)

        # TODO: brake if within brake_deadband?
        if delta_throttle > 0.:
            throttle = delta_throttle
        elif delta_throttle < 0:
            throttle = 0.
            delta_throttle = max(delta_throttle, self.decel_limit)
            brake = abs(delta_throttle) * self.vehicle_mass * self.wheel_radius # torque N*m
        else:
            throttle = 0.

        if proposed_linear_vel == 0.0:
            # complete standstill
            brake = 400
            throttle = 0
            self.reset()

        # Steering Controller
        steering = self.yaw_controller.get_steering(proposed_linear_vel, proposed_angular_vel, current_linear_vel)

        return throttle, brake, steering
    
    def reset(self):
        self.vel_pid.reset()
