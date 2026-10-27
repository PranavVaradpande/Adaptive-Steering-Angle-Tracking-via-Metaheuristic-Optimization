#!/usr/bin/env python3
import rospy
import math
import time
import numpy as np
from std_msgs.msg import Float32MultiArray
from pymodbus.client import ModbusTcpClient as ModbusClient

# -------------------------------------------------------
#                 MODBUS CONFIGURATION
# -------------------------------------------------------
PLC_HOST = '192.168.140.5'
PLC_PORT = 502
UNIT = 0x1

client = ModbusClient(PLC_HOST, PLC_PORT)
if not client.connect():
    print("❌ Modbus Connection Failed!")
    exit()
else:
    print("✅ Connected to PLC")

# -------------------------------------------------------
#                    GLOBAL VARIABLES
# -------------------------------------------------------
steering_center = 535
current_angle = 0
current_x, current_y = 0, 0
heading = 0
previous_velocity = 0

client.write_registers(22, 3000, unit=UNIT)
client.write_registers(17, steering_center, unit=UNIT)

# -------------------------------------------------------
#                PSO-PI TUNER CLASS
# -------------------------------------------------------
class PSOPITuner:
    def __init__(self, num_particles=100, pso_iterations=4):
        self.num_particles = num_particles
        self.pso_iterations = pso_iterations

        # Physical Plant Constants from GWO_PI.txt
        self.Ieq = 0.428
        self.Beq = 4.28
        self.T_s = 0.01
        self.lambda1 = 0.70
        self.lambda2 = 0.30

        # Parameter Bounds
        self.Kp_min, self.Kp_max = 20.0, 100.0
        self.Ki_min, self.Ki_max = 0.001, 0.0050
        self.vmax_Kp, self.vmax_Ki = 10.0, 0.001
        self.Max_Torque = 40.0

        self.c1, self.c2 = 2.0, 2.0
        self.Wmax, self.Wmin = 0.9, 0.1
        self.delt = 0.1

        self.gBest_Kp = 40.0
        self.gBest_Ki = 0.002
        self.gBest_Fitness = 1e36

        # Swarm Matrices
        self.pos_Kp = np.random.uniform(self.Kp_min, self.Kp_max, self.num_particles)
        self.pos_Ki = np.random.uniform(self.Ki_min, self.Ki_max, self.num_particles)
        self.vel_Kp = np.zeros(self.num_particles)
        self.vel_Ki = np.zeros(self.num_particles)

        self.pBest_Kp = np.copy(self.pos_Kp)
        self.pBest_Ki = np.copy(self.pos_Ki)
        self.pBest_Fitness = np.full(self.num_particles, 1e36)

    def optimize(self, target_angle, actual_angle, error, e_prev, u_k, steering_torque, mu_friction, vehicle_speed):
        actual_vel = (actual_angle - self.Ieq) / self.T_s
        calc_disturbance = steering_torque + (mu_friction * 10.0 * (actual_angle / (vehicle_speed + 0.1)))

        # Warm Start Particle 0
        self.pos_Kp[0] = self.gBest_Kp
        self.pos_Ki[0] = self.gBest_Ki

        for k in range(1, self.pso_iterations + 1):
            w = self.Wmax - (k * (self.Wmax - self.Wmin) / self.pso_iterations)

            for i in range(self.num_particles):
                delta_u = self.pos_Kp[i] * (error - e_prev) + self.pos_Ki[i] * error
                u_new = np.clip(u_k + delta_u, -self.Max_Torque, self.Max_Torque)

                pred_accel = (u_new - calc_disturbance - (self.Beq * (actual_vel / 57.29578))) / self.Ieq
                pred_vel = (actual_vel / 57.29578) + (pred_accel * self.T_s)
                pred_angle = actual_angle + (pred_vel * self.T_s * 57.29578)

                J = self.lambda1 * abs(target_angle - pred_angle) + self.lambda2 * abs(delta_u)

                if J < self.pBest_Fitness[i]:
                    self.pBest_Fitness[i] = J
                    self.pBest_Kp[i] = self.pos_Kp[i]
                    self.pBest_Ki[i] = self.pos_Ki[i]

                if self.pBest_Fitness[i] < self.gBest_Fitness:
                    self.gBest_Fitness = self.pBest_Fitness[i]
                    self.gBest_Kp = self.pBest_Kp[i]
                    self.gBest_Ki = self.pBest_Ki[i]

            # Swarm Velocity and Position update
            r1, r2 = np.random.rand(2, self.num_particles)
            self.vel_Kp = w * self.vel_Kp + self.c1 * r1 * (self.pBest_Kp - self.pos_Kp) + self.c2 * r2 * (self.gBest_Kp - self.pos_Kp)
            self.vel_Ki = w * self.vel_Ki + self.c1 * r1 * (self.pBest_Ki - self.pos_Ki) + self.c2 * r2 * (self.gBest_Ki - self.pos_Ki)

            self.vel_Kp = np.clip(self.vel_Kp, -self.vmax_Kp, self.vmax_Kp)
            self.vel_Ki = np.clip(self.vel_Ki, -self.vmax_Ki, self.vmax_Ki)

            self.pos_Kp = np.clip(self.pos_Kp + self.vel_Kp * self.delt, self.Kp_min, self.Kp_max)
            self.pos_Ki = np.clip(self.pos_Ki + self.vel_Ki * self.delt, self.Ki_min, self.Ki_max)

        return self.gBest_Kp, self.gBest_Ki

# -------------------------------------------------------
#               INCREMENTAL PI CONTROLLER
# -------------------------------------------------------
class IncrementalPIController:
    def __init__(self, u_max=450.0):
        self.e_k1 = 0.0
        self.u_k1 = 0.0
        self.u_max = u_max

    def compute(self, e_k, Kp, Ki):
        delta_u_k = Kp * (e_k - self.e_k1) + Ki * e_k
        delta_u_k = np.clip(delta_u_k, -5.0, 5.0)

        u_k = np.clip(self.u_k1 + delta_u_k, -self.u_max, self.u_max)

        self.e_k1 = e_k
        self.u_k1 = u_k
        return u_k, delta_u_k

# -------------------------------------------------------
#                   LOGGING FUNCTION
# -------------------------------------------------------
LOG_FILE = "/home/tihan/Downloads/navigation_log_pi.txt"

def log_parameters(current_x, current_y, target_x, target_y, off_x, off_y, heading, bearing, bearing_diff, steer_output, velocity):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"Time: {timestamp}, Current:({current_x:.3f}, {current_y:.3f}), Target:({target_x:.3f}, {target_y:.3f}), Offset:({off_x:.3f}, {off_y:.3f}), Heading:{heading:.2f}, Bearing:{bearing:.2f}, BearingDiff:{bearing_diff:.2f}, Steer:{steer_output}, Velocity:{velocity:.2f}\n")

# -------------------------------------------------------
#              WAYPOINT FILE READER
# -------------------------------------------------------
def extract_waypoints_from_file(file_path):
    waypoints = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().replace("[", "").replace("]", "").replace(",", " ")
                parts = line.split()
                if len(parts) >= 2:
                    waypoints.append([float(parts[0]), float(parts[1])])
    except Exception as e:
        print("❌ Error reading waypoint file:", e)
    print("✅ Loaded", len(waypoints), "waypoints")
    return waypoints

# -------------------------------------------------------
#                   PLC FUNCTIONS
# -------------------------------------------------------
def read_angle():
    val = client.read_holding_registers(1, 1, unit=UNIT)
    return val.registers[0] if val else 0

def set_steer(angle):
    direction = angle < 0
    client.write_coil(20, direction, unit=UNIT)
    client.write_registers(400, abs(int(angle)), unit=UNIT)

def accelerate(value):
    client.write_registers(500, value, unit=UNIT)

def set_drive_mode(forward=False, reverse=False):
    client.write_coil(502, forward, unit=UNIT)
    client.write_coil(504, reverse, unit=UNIT)

# -------------------------------------------------------
#                ROS CALLBACKS
# -------------------------------------------------------
def callback_ndt_pose(data):
    global current_x, current_y
    current_x, current_y = data.data[0], data.data[1]

def callback_eular_angle(data):
    global heading
    heading = data.data[2] * (180 / np.pi)

def find_nearest_waypoint_index(waypoints):
    min_distance = float("inf")
    nearest_index = 0
    for i, (wx, wy) in enumerate(waypoints):
        dist = math.hypot(wx - current_x, wy - current_y)
        if dist < min_distance:
            min_distance = dist
            nearest_index = i
    return nearest_index

def fun_bearing_diff(required_bearing):
    bearing_diff = -heading + required_bearing
    if bearing_diff < -180: bearing_diff += 360
    if bearing_diff > 180: bearing_diff -= 360
    return bearing_diff

# -------------------------------------------------------
#                    ROS SETUP
# -------------------------------------------------------
rospy.init_node("gwo_pi_subscriber")
rospy.Subscriber("/radar_odom_mod", Float32MultiArray, callback_ndt_pose)
rospy.Subscriber("/radar_odom_eular", Float32MultiArray, callback_eular_angle)

waypoints = extract_waypoints_from_file("/home/tihan/Downloads/mmlab.txt")
wp = find_nearest_waypoint_index(waypoints)

# Instantiate PSO and Incremental PI
pso_tuner = PSOPITuner(num_particles=100, pso_iterations=4)
pi_controller = IncrementalPIController(u_max=450.0)

# -------------------------------------------------------
#                   CONTROL LOOP
# -------------------------------------------------------
while not rospy.is_shutdown():
    target_x, target_y = waypoints[wp]
    distance = math.hypot(current_x - target_x, current_y - target_y)

    off_x = target_x - current_x
    off_y = target_y - current_y
    bearing = math.degrees(math.atan2(off_y, off_x)) % 360
    bearing_diff = fun_bearing_diff(bearing)

    # ------------ VELOCITY CONTROL (MAX 10 KM/H) ------------
    set_drive_mode(forward=True)
    target_vel = 10.0  # Max limit set to 10 km/h
    current_vel = min(previous_velocity + 0.5, target_vel)
    previous_velocity = current_vel
    accelerate(int(current_vel))

    # ------------ GWO/PSO INCREMENTAL PI STEERING -----------
    actual_angle = read_angle()
    error = bearing_diff - actual_angle

    # Run PSO Tuner to adapt Kp and Ki dynamically
    Kp, Ki = pso_tuner.optimize(
        target_angle=bearing_diff,
        actual_angle=actual_angle,
        error=error,
        e_prev=pi_controller.e_k1,
        u_k=pi_controller.u_k1,
        steering_torque=0.0,
        mu_friction=0.8,
        vehicle_speed=current_vel / 3.6
    )

    # Calculate Incremental Control Output
    steer_output, delta_u = pi_controller.compute(error, Kp, Ki)
    set_steer(steer_output)

    print(f"WP: {wp} | Dist: {round(distance, 2)} | Vel: {round(current_vel, 2)} | Steer: {int(steer_output)} | Kp: {round(Kp,2)} Ki: {round(Ki,5)}")

    log_parameters(current_x, current_y, target_x, target_y, off_x, off_y, heading, bearing, bearing_diff, steer_output, current_vel)

    if distance < 0.5 and wp < len(waypoints) - 1:
        wp += 1

    time.sleep(0.01)