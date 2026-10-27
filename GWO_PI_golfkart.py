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

# Initial PLC Setup
client.write_registers(22, 3000, unit=UNIT)
client.write_registers(17, steering_center, unit=UNIT)

# -------------------------------------------------------
#           PSO INCREMENTAL PI CONTROLLER CLASS
# -------------------------------------------------------
class PSO_Incremental_PI:
    """
    Online Particle Swarm Optimization (PSO) Tuner for Incremental PI Control.
    Configured with 100 particles and 4 optimization iterations per step.
    """
    def __init__(self, num_particles=100, pso_iterations=4, ts=0.01):
        self.num_particles = num_particles
        self.pso_iterations = pso_iterations
        self.ts = ts

        # Physical Plant Constants (from CODESYS plant model)
        self.Ieq = 0.428
        self.Beq = 4.28

        # Standard PSO Search & Performance Weights
        self.Wmax = 0.9
        self.Wmin = 0.1
        self.c1 = 2.0
        self.c2 = 2.0
        self.delt = 0.1
        self.lambda1 = 0.70
        self.lambda2 = 0.30

        # Search Space & Acceleration Bounds
        self.Kp_min, self.Kp_max = 20.0, 100.0
        self.Ki_min, self.Ki_max = 0.001, 0.0050
        self.vmax_Kp = 10.0
        self.vmax_Ki = 0.001
        self.Max_Torque = 40.0
        self.U_MAX = 40.0

        # Swarm Matrices
        self.pos_Kp = np.random.uniform(self.Kp_min, self.Kp_max, self.num_particles)
        self.pos_Ki = np.random.uniform(self.Ki_min, self.Ki_max, self.num_particles)
        self.vel_Kp = np.zeros(self.num_particles)
        self.vel_Ki = np.zeros(self.num_particles)

        self.pBest_Kp = np.copy(self.pos_Kp)
        self.pBest_Ki = np.copy(self.pos_Ki)
        self.pBest_Fitness = np.full(self.num_particles, 1.0e36)

        # Global Best Initialization
        self.gBest_Kp = 40.0
        self.gBest_Ki = 0.002
        self.gBest_Fitness = 1.0e36

        # Internal State Registers
        self.e_prev = 0.0
        self.u_prev = 0.0
        self.prev_actual_angle = 0.0

    def compute_control(self, target_angle, actual_angle, steering_torque=0.0, mu_friction=0.8, vehicle_speed=2.78):
        error = target_angle - actual_angle
        actual_vel = (actual_angle - self.prev_actual_angle) / self.ts
        calc_disturbance = steering_torque + (mu_friction * 10.0 * (actual_angle / (vehicle_speed + 0.1)))

        # Online Re-initialization & Warm Start
        self.pos_Kp = np.random.uniform(self.Kp_min, self.Kp_max, self.num_particles)
        self.pos_Ki = np.random.uniform(self.Ki_min, self.Ki_max, self.num_particles)
        self.vel_Kp.fill(0.0)
        self.vel_Ki.fill(0.0)

        # Warm start Particle 0 with last known global best
        self.pos_Kp[0] = self.gBest_Kp
        self.pos_Ki[0] = self.gBest_Ki

        self.pBest_Kp = np.copy(self.pos_Kp)
        self.pBest_Ki = np.copy(self.pos_Ki)
        self.pBest_Fitness.fill(1.0e36)
        self.gBest_Fitness = 1.0e36

        # Step 1: Initial Evaluation of Personal Bests
        for i in range(self.num_particles):
            delta_u = self.pBest_Kp[i] * (error - self.e_prev) + self.pBest_Ki[i] * error
            u_new = np.clip(self.u_prev + delta_u, -self.Max_Torque, self.Max_Torque)

            pred_accel = (u_new - calc_disturbance - (self.Beq * (actual_vel / 57.29578))) / self.Ieq
            pred_vel = (actual_vel / 57.29578) + (pred_accel * self.ts)
            pred_angle = actual_angle + (pred_vel * self.ts * 57.29578)

            J = self.lambda1 * abs(target_angle - pred_angle) + self.lambda2 * abs(delta_u)
            self.pBest_Fitness[i] = J

            if J < self.gBest_Fitness:
                self.gBest_Fitness = J
                self.gBest_Kp = self.pBest_Kp[i]
                self.gBest_Ki = self.pBest_Ki[i]

        # Step 2: PSO Optimization Runs (4 Iterations)
        for k in range(1, self.pso_iterations + 1):
            w = self.Wmax - (k * (self.Wmax - self.Wmin) / self.pso_iterations)

            for i in range(self.num_particles):
                delta_u = self.pos_Kp[i] * (error - self.e_prev) + self.pos_Ki[i] * error
                u_new = np.clip(self.u_prev + delta_u, -self.Max_Torque, self.Max_Torque)

                pred_accel = (u_new - calc_disturbance - (self.Beq * (actual_vel / 57.29578))) / self.Ieq
                pred_vel = (actual_vel / 57.29578) + (pred_accel * self.ts)
                pred_angle = actual_angle + (pred_vel * self.ts * 57.29578)

                J = self.lambda1 * abs(target_angle - pred_angle) + self.lambda2 * abs(delta_u)

                if J < self.pBest_Fitness[i]:
                    self.pBest_Fitness[i] = J
                    self.pBest_Kp[i] = self.pos_Kp[i]
                    self.pBest_Ki[i] = self.pos_Ki[i]

                if J < self.gBest_Fitness:
                    self.gBest_Fitness = J
                    self.gBest_Kp = self.pos_Kp[i]
                    self.gBest_Ki = self.pos_Ki[i]

            # Velocity & Position Update Equations
            r1 = np.random.rand(self.num_particles)
            r2 = np.random.rand(self.num_particles)

            self.vel_Kp = w * self.vel_Kp + self.c1 * r1 * (self.pBest_Kp - self.pos_Kp) + self.c2 * r2 * (self.gBest_Kp - self.pos_Kp)
            self.vel_Ki = w * self.vel_Ki + self.c1 * r1 * (self.pBest_Ki - self.pos_Ki) + self.c2 * r2 * (self.gBest_Ki - self.pos_Ki)

            # Hardwall Velocity Clamping
            self.vel_Kp = np.clip(self.vel_Kp, -self.vmax_Kp, self.vmax_Kp)
            self.vel_Ki = np.clip(self.vel_Ki, -self.vmax_Ki, self.vmax_Ki)

            # Position Update with Interval Step
            self.pos_Kp = np.clip(self.pos_Kp + self.vel_Kp * self.delt, self.Kp_min, self.Kp_max)
            self.pos_Ki = np.clip(self.pos_Ki + self.vel_Ki * self.delt, self.Ki_min, self.Ki_max)

        # Step 3: Incremental PI Control Equation Execution
        delta_u_k = self.gBest_Kp * (error - self.e_prev) + self.gBest_Ki * error
        delta_u_k = np.clip(delta_u_k, -5.0, 5.0)

        u_k = np.clip(self.u_prev + delta_u_k, -self.U_MAX, self.U_MAX)

        # Update Controller Registers
        self.e_prev = error
        self.u_prev = u_k
        self.prev_actual_angle = actual_angle

        return u_k, self.gBest_Kp, self.gBest_Ki

# -------------------------------------------------------
#                   LOGGING FUNCTION
# -------------------------------------------------------
LOG_FILE = "/home/tihan/Downloads/navigation_log_pso_pi.txt"

def log_parameters(current_x, current_y, target_x, target_y,
                   off_x, off_y, heading, bearing, bearing_diff,
                   steer_output, velocity, kp, ki):

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a") as f:
        f.write(
            f"Time: {timestamp}, "
            f"Current:({current_x:.3f}, {current_y:.3f}), "
            f"Target:({target_x:.3f}, {target_y:.3f}), "
            f"Offset:({off_x:.3f}, {off_y:.3f}), "
            f"Heading:{heading:.2f}, "
            f"Bearing:{bearing:.2f}, "
            f"BearingDiff:{bearing_diff:.2f}, "
            f"Steer:{steer_output}, "
            f"Velocity:{velocity:.2f}, "
            f"Kp:{kp:.4f}, Ki:{ki:.6f}\n"
        )

# -------------------------------------------------------
#              WAYPOINT FILE READER
# -------------------------------------------------------
def extract_waypoints_from_file(file_path):
    waypoints = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                line = line.replace("[", "").replace("]", "").replace(",", " ")
                parts = line.split()

                if len(parts) < 2:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                waypoints.append([x, y])

    except Exception as e:
        print("❌ Error reading waypoint file:", e)

    print("✅ Loaded", len(waypoints), "waypoints")
    return waypoints

# -------------------------------------------------------
#                   PLC FUNCTIONS
# -------------------------------------------------------
def read_angle():
    global current_angle
    val = client.read_holding_registers(1, 1, unit=UNIT)
    if val:
        current_angle = val.registers[0]
    return current_angle

def set_steer(angle):
    direction = angle < 0
    client.write_coil(20, direction, unit=UNIT)
    client.write_registers(400, abs(int(angle)), unit=UNIT)

def accelerate(value):
    client.write_registers(500, int(value), unit=UNIT)

def set_drive_mode(forward=False, reverse=False):
    client.write_coil(502, forward, unit=UNIT)
    client.write_coil(504, reverse, unit=UNIT)

# -------------------------------------------------------
#                ROS CALLBACKS
# -------------------------------------------------------
def callback_ndt_pose(data):
    global current_x, current_y
    current_x = data.data[0]
    current_y = data.data[1]

def callback_eular_angle(data):
    global heading
    heading = data.data[2] * (180 / np.pi)

# -------------------------------------------------------
#              NEAREST WAYPOINT
# -------------------------------------------------------
def find_nearest_waypoint_index(waypoints):
    min_distance = float("inf")
    nearest_index = 0

    for i, (wx, wy) in enumerate(waypoints):
        dist = math.hypot(wx - current_x, wy - current_y)
        if dist < min_distance:
            min_distance = dist
            nearest_index = i

    print("🚩 Nearest WP =", nearest_index)
    return nearest_index

# -------------------------------------------------------
#                BEARING DIFFERENCE
# -------------------------------------------------------
def fun_bearing_diff(required_bearing):
    bearing_diff = -heading + required_bearing
    if bearing_diff < -180:
        bearing_diff += 360
    if bearing_diff > 180:
        bearing_diff -= 360
    return bearing_diff

# -------------------------------------------------------
#                    ROS SETUP
# -------------------------------------------------------
def listener():
    rospy.init_node("topic_subscriber")
    rospy.Subscriber("/radar_odom_mod", Float32MultiArray, callback_ndt_pose)
    rospy.Subscriber("/radar_odom_eular", Float32MultiArray, callback_eular_angle)

listener()

# -------------------------------------------------------
#                  LOAD WAYPOINTS
# -------------------------------------------------------
waypoints = extract_waypoints_from_file("/home/tihan/Downloads/mmlab.txt")
wp = find_nearest_waypoint_index(waypoints)

# Controller Instantiation (100 Particles, 4 Iterations)
controller = PSO_Incremental_PI(num_particles=100, pso_iterations=4)

# -------------------------------------------------------
#                   CONTROL LOOP
# -------------------------------------------------------
while not rospy.is_shutdown():

    target_x, target_y = waypoints[wp]
    distance = math.hypot(current_x - target_x, current_y - target_y)

    # -------------------- BEARING -----------------------
    off_x = target_x - current_x
    off_y = target_y - current_y

    bearing = math.degrees(math.atan2(off_y, off_x)) % 360
    bearing_diff = fun_bearing_diff(bearing)

    # -------------------- VELOCITY (MAX 10 KM/H) --------
    set_drive_mode(forward=True)

    target_vel = 10.0  # Speed restricted to maximum of 10 km/h
    current_vel = min(previous_velocity + 0.5, target_vel)
    previous_velocity = current_vel
    accelerate(current_vel)

    # -------------------- STEERING ----------------------
    desired_angle = 40.0 * np.arctan(1 * 2 * 3.5 * math.sin(np.radians(bearing_diff)) / 8.0)
    actual_steering_angle = float(read_angle() - steering_center)

    u_steer, Kp_opt, Ki_opt = controller.compute_control(
        target_angle=desired_angle,
        actual_angle=actual_steering_angle,
        steering_torque=0.0,
        mu_friction=0.8,
        vehicle_speed=current_vel / 3.6
    )

    steer_output = int(np.clip(u_steer, -40, 40))
    set_steer(steer_output)

    # -------------------- PRINT -------------------------
    print(
        f"WP: {wp} | Dist: {round(distance, 2)} | Vel: {round(current_vel, 2)} "
        f"| Steer: {steer_output} | Kp: {round(Kp_opt, 3)} | Ki: {round(Ki_opt, 5)}"
    )

    # -------------------- LOG TO FILE -------------------
    log_parameters(
        current_x, current_y,
        target_x, target_y,
        off_x, off_y,
        heading, bearing, bearing_diff,
        steer_output, current_vel,
        Kp_opt, Ki_opt
    )

    # -------------------- NEXT WAYPOINT -----------------
    if distance < 0.5 and wp < len(waypoints) - 1:
        wp += 1

    time.sleep(0.1)