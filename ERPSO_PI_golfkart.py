#!/usr/bin/env python3
from pymodbus.client import ModbusTcpClient as ModbusClient
import rospy
import math
import time
import numpy as np
from std_msgs.msg import Float32MultiArray

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

# Initial Values
client.write_registers(22, 3000, unit=UNIT)
client.write_registers(17, steering_center, unit=UNIT)

# -------------------------------------------------------
#            ERPSO INCREMENTAL PI CONTROLLER
# -------------------------------------------------------
class ERPSO_Incremental_PI:
    def __init__(self, num_particles=100, pso_iterations=4, ts=0.01):
        self.num_particles = num_particles
        self.pso_iterations = pso_iterations
        self.ts = ts

        # Physical Plant Constants
        self.Ieq = 0.428
        self.Beq = 4.28

        # Standard PSO & Cost Weights
        self.Wmax = 0.9
        self.Wmin = 0.1
        self.delt = 0.1
        self.lambda1 = 0.70
        self.lambda2 = 0.30

        # Search Bounds
        self.Kp_min, self.Kp_max = 20.0, 100.0
        self.Ki_min, self.Ki_max = 0.001, 0.0050
        self.vmax_Kp = 10.0
        self.vmax_Ki = 0.001
        self.Max_Torque = 1.0
        self.U_MAX = 40.0

        # Grid & Entropy Configuration (32x32 = 1024 cells)
        self.GridSize = 32
        self.Hmax = 6.9314

        # Global Best Initialization
        self.gBest_Kp = 40.0
        self.gBest_Ki = 0.002
        self.gBest_Fitness = 1.0e36

        # Internal State Registers
        self.e_prev = 0.0
        self.u_prev = 0.0
        self.prev_actual_angle = 0.0

    def compute_control(self, target_angle, actual_angle, steering_torque=0.0, mu_friction=0.8, vehicle_speed=2.78, lambda_entropy=0.1):
        error = target_angle - actual_angle
        actual_vel = (actual_angle - self.prev_actual_angle) / self.ts
        calc_disturbance = steering_torque + (mu_friction * 10.0 * (actual_angle / (vehicle_speed + 0.1)))

        # Reseed Particle Swarm
        pos_Kp = np.random.uniform(self.Kp_min, self.Kp_max, self.num_particles)
        pos_Ki = np.random.uniform(self.Ki_min, self.Ki_max, self.num_particles)
        
        # Particle 1 warm-start with global best
        pos_Kp[0] = self.gBest_Kp
        pos_Ki[0] = self.gBest_Ki

        vel_Kp = np.zeros(self.num_particles)
        vel_Ki = np.zeros(self.num_particles)

        pBest_Kp = np.copy(pos_Kp)
        pBest_Ki = np.copy(pos_Ki)
        pBest_Fitness = np.full(self.num_particles, 1.0e36)

        gBest_Fitness = 1.0e36
        gBest_Kp = self.gBest_Kp
        gBest_Ki = self.gBest_Ki

        # Unified ERPSO Iteration Loop
        for k in range(self.pso_iterations + 1):
            raw_J = np.zeros(self.num_particles)
            grid_counts = np.zeros(1024, dtype=int)
            particle_cell_idx = np.zeros(self.num_particles, dtype=int)

            # 1. Evaluate Raw Fitness
            for i in range(self.num_particles):
                delta_u = pos_Kp[i] * (error - self.e_prev) + pos_Ki[i] * error
                u_unclamped = self.u_prev + delta_u
                u_new = np.clip(u_unclamped, -self.Max_Torque, self.Max_Torque)

                pred_accel = (u_new - calc_disturbance - (self.Beq * (actual_vel / 57.29578))) / self.Ieq
                pred_vel = (actual_vel / 57.29578) + (pred_accel * self.ts)
                pred_angle = actual_angle + (pred_vel * self.ts * 57.29578)

                raw_J[i] = self.lambda1 * abs(target_angle - pred_angle) + self.lambda2 * abs(delta_u)

            # 2. 32x32 Cell Spatial Grid Mapping
            for i in range(self.num_particles):
                gx = int(np.clip(((pos_Kp[i] - self.Kp_min) / (self.Kp_max - self.Kp_min)) * 32, 0, 31))
                gy = int(np.clip(((pos_Ki[i] - self.Ki_min) / (self.Ki_max - self.Ki_min)) * 32, 0, 31))
                cell = gy * self.GridSize + gx
                particle_cell_idx[i] = cell
                grid_counts[cell] += 1

            # Compute Shannon Entropy
            entropy = 0.0
            for count in grid_counts:
                if count > 0:
                    prob = count / float(self.num_particles)
                    entropy -= prob * prob

            entropy_norm = np.clip(entropy / self.Hmax, 0.0, 1.0)

            # Adaptive Inertia & Cognitive/Social Acceleration Coefficients
            w = self.Wmax - (self.Wmax - self.Wmin) * entropy_norm
            c1 = 2.0 + 0.3 * (1.0 - entropy_norm)
            c2 = 2.0 - 0.3 * (1.0 - entropy_norm)

            # 3. Entropy Regularization & Best Updates
            for i in range(self.num_particles):
                count = grid_counts[particle_cell_idx[i]]
                H_i = -math.log(float(count) / self.num_particles) if count > 0 else 0.0
                JER = raw_J[i] * (1.0 - lambda_entropy * H_i)

                if JER < pBest_Fitness[i]:
                    pBest_Fitness[i] = JER
                    pBest_Kp[i] = pos_Kp[i]
                    pBest_Ki[i] = pos_Ki[i]

                if pBest_Fitness[i] < gBest_Fitness:
                    gBest_Fitness = pBest_Fitness[i]
                    gBest_Kp = pBest_Kp[i]
                    gBest_Ki = pBest_Ki[i]

            # 4. Swarm Velocity and Position Step
            if k < self.pso_iterations:
                r1 = np.random.rand(self.num_particles)
                r2 = np.random.rand(self.num_particles)

                vel_Kp = w * vel_Kp + c1 * r1 * (pBest_Kp - pos_Kp) + c2 * r2 * (gBest_Kp - pos_Kp)
                vel_Ki = w * vel_Ki + c1 * r1 * (pBest_Ki - pos_Ki) + c2 * r2 * (gBest_Ki - pos_Ki)

                vel_Kp = np.clip(vel_Kp, -self.vmax_Kp, self.vmax_Kp)
                vel_Ki = np.clip(vel_Ki, -self.vmax_Ki, self.vmax_Ki)

                pos_Kp = np.clip(pos_Kp + vel_Kp * self.delt, self.Kp_min, self.Kp_max)
                pos_Ki = np.clip(pos_Ki + vel_Ki * self.delt, self.Ki_min, self.Ki_max)

        self.gBest_Kp = gBest_Kp
        self.gBest_Ki = gBest_Ki

        # Executing Incremental PI Equation
        delta_u_k = self.gBest_Kp * (error - self.e_prev) + self.gBest_Ki * error
        delta_u_k = np.clip(delta_u_k, -5.0, 5.0)

        u_k = np.clip(self.u_prev + delta_u_k, -self.U_MAX, self.U_MAX)

        # Update Memory
        self.e_prev = error
        self.u_prev = u_k
        self.prev_actual_angle = actual_angle

        return u_k, self.gBest_Kp, self.gBest_Ki

# -------------------------------------------------------
#                   LOGGING FUNCTION
# -------------------------------------------------------
LOG_FILE = "/home/tihan/Downloads/navigation_log_erpso_pi.txt"

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
controller = ERPSO_Incremental_PI(num_particles=100, pso_iterations=4)

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

    target_vel = 10.0  # Capped at 10 km/h
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