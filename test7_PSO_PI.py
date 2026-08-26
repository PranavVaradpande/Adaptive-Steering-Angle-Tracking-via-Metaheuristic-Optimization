from pymodbus.client import ModbusTcpClient as ModbusClient
import rospy
import math
import time
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import signal
import sys

# =========================================================
# PSO_PI CONTROLLER CLASSES (Fixed)
# =========================================================

class PSO_Tuner:
    def __init__(self):
        self.Ieq = 0.428
        self.Beq = 4.28
        self.NumParticles = 20
        self.PSO_Iterations = 4
        self.c1 = 2.0
        self.c2 = 2.0
        self.Wmax = 0.9
        self.Wmin = 0.1
        self.w = 0.0
        self.delt = 0.1
        self.J_max = 0.0
        self.J_min = 0.0
        self.lambda1 = 0.70
        self.lambda2 = 0.30
        self.Kp_min = 20.0
        self.Kp_max = 100.0
        self.Ki_min = 0.001
        self.Ki_max = 0.0050
        self.vmax_Kp = 10.0
        self.vmax_Ki = 0.001
        self.Max_Torque = 40.0

        self.pos_Kp = [0.0] * self.NumParticles
        self.pos_Ki = [0.0] * self.NumParticles
        self.vel_Kp = [0.0] * self.NumParticles
        self.vel_Ki = [0.0] * self.NumParticles
        self.pBest_Kp = [0.0] * self.NumParticles
        self.pBest_Ki = [0.0] * self.NumParticles
        self.pBest_Fitness = [0.0] * self.NumParticles

        self.gBest_Kp = 40.0
        self.gBest_Ki = 0.002
        self.gBest_Fitness = 1.0E36

        self.delta_u = 0.0
        self.u_unclamped = 0.0
        self.u_new = 0.0
        self.J = 0.0
        self.Actual_Vel = 0.0
        self.Prev_Actual_Angle = 0.0
        self.Calculated_Disturbance = 0.0
        self.Pred_Accel = 0.0
        self.Pred_Vel = 0.0
        self.Pred_Angle = 0.0
        self.fitness_sum = 0.0
        self.fitness_var = 0.0
        self.center_Kp = 0.0
        self.center_Ki = 0.0
        self.temp_dist = 0.0
        self.seed = 12345
        self.r1 = 0.0
        self.r2 = 0.0
        self.e_prev = 0.0
        self.initialized = False

        self.Kp = 40.0
        self.Ki = 0.002
        self.Global_Fitness = 1.0E36
        self.Mean_Fitness = 0.0
        self.Fitness_Diversity_Index = 0.0
        self.Swarm_Radius = 0.0
        self.Iteration_Count = 0
        self.Fitness_Evaluations = 0
        self.Convergence_Cost = 0.0

    def _lcg_random(self):
        self.seed = (self.seed * 1103515245 + 12345) & 0xFFFFFFFF
        return float(self.seed & 0x7FFFFFFF) / 2147483647.0

    def run(self, Enable, Target_Angle, Actual_Angle, Error, U_k, Mu_friction, T_s=0.1):
        if not self.initialized:
            self.Kp = self.gBest_Kp
            self.Ki = self.gBest_Ki
            self.e_prev = Error
            self.Prev_Actual_Angle = Actual_Angle
            self.initialized = True

        if Enable:
            self.Actual_Vel = (Actual_Angle - self.Prev_Actual_Angle) / T_s
            self.gBest_Fitness = 1.0E36

            for i in range(self.NumParticles):
                self.pos_Kp[i] = self.Kp_min + (self.Kp_max - self.Kp_min) * self._lcg_random()
                self.pos_Ki[i] = self.Ki_min + (self.Ki_max - self.Ki_min) * self._lcg_random()
                self.vel_Kp[i] = 0.0
                self.vel_Ki[i] = 0.0
                self.pBest_Kp[i] = self.pos_Kp[i]
                self.pBest_Ki[i] = self.pos_Ki[i]
                self.pBest_Fitness[i] = 1.0E36

            self.pos_Kp[0] = self.Kp
            self.pos_Ki[0] = self.Ki
            self.pBest_Kp[0] = self.Kp
            self.pBest_Ki[0] = self.Ki

            for i in range(self.NumParticles):
                self.delta_u = self.pBest_Kp[i] * (Error - self.e_prev) + self.pBest_Ki[i] * Error
                self.u_unclamped = U_k + self.delta_u
                self.u_new = max(-self.Max_Torque, min(self.Max_Torque, self.u_unclamped))

                # FIXED: removed extra parenthesis after Mu_friction*self.u_new
                self.Pred_Accel = (Mu_friction * self.u_new - (self.Beq * (self.Actual_Vel / 57.29578))) / self.Ieq
                self.Pred_Vel = (self.Actual_Vel / 57.29578) + (self.Pred_Accel * T_s)
                self.Pred_Angle = Actual_Angle + (self.Pred_Vel * T_s * 57.29578)

                self.J = self.lambda1 * abs(Target_Angle - self.Pred_Angle) + self.lambda2 * abs(self.delta_u)
                self.pBest_Fitness[i] = self.J

                if self.pBest_Fitness[i] < self.gBest_Fitness:
                    self.gBest_Fitness = self.pBest_Fitness[i]
                    self.gBest_Kp = self.pBest_Kp[i]
                    self.gBest_Ki = self.pBest_Ki[i]

            for k in range(1, self.PSO_Iterations + 1):
                self.w = self.Wmax - (float(k) * (self.Wmax - self.Wmin) / float(self.PSO_Iterations))

                for i in range(self.NumParticles):
                    self.delta_u = self.pos_Kp[i] * (Error - self.e_prev) + self.pos_Ki[i] * Error
                    self.u_unclamped = U_k + self.delta_u
                    self.u_new = max(-self.Max_Torque, min(self.Max_Torque, self.u_unclamped))

                    self.Pred_Accel = (Mu_friction * self.u_new - (self.Beq * (self.Actual_Vel / 57.29578))) / self.Ieq
                    self.Pred_Vel = (self.Actual_Vel / 57.29578) + (self.Pred_Accel * T_s)
                    self.Pred_Angle = Actual_Angle + (self.Pred_Vel * T_s * 57.29578)

                    self.J = self.lambda1 * abs(Target_Angle - self.Pred_Angle) + self.lambda2 * abs(self.delta_u)

                    if self.J < self.pBest_Fitness[i]:
                        self.pBest_Fitness[i] = self.J
                        self.pBest_Kp[i] = self.pos_Kp[i]
                        self.pBest_Ki[i] = self.pos_Ki[i]

                    if self.pBest_Fitness[i] < self.gBest_Fitness:
                        self.gBest_Fitness = self.pBest_Fitness[i]
                        self.gBest_Kp = self.pBest_Kp[i]
                        self.gBest_Ki = self.pBest_Ki[i]

                for i in range(self.NumParticles):
                    self.r1 = self._lcg_random()
                    self.r2 = self._lcg_random()

                    self.vel_Kp[i] = (self.w * self.vel_Kp[i] +
                                      self.c1 * self.r1 * (self.pBest_Kp[i] - self.pos_Kp[i]) +
                                      self.c2 * self.r2 * (self.gBest_Kp - self.pos_Kp[i]))
                    self.vel_Ki[i] = (self.w * self.vel_Ki[i] +
                                      self.c1 * self.r1 * (self.pBest_Ki[i] - self.pos_Ki[i]) +
                                      self.c2 * self.r2 * (self.gBest_Ki - self.pos_Ki[i]))

                    self.vel_Kp[i] = max(-self.vmax_Kp, min(self.vmax_Kp, self.vel_Kp[i]))
                    self.vel_Ki[i] = max(-self.vmax_Ki, min(self.vmax_Ki, self.vel_Ki[i]))

                    self.pos_Kp[i] += self.vel_Kp[i] * self.delt
                    self.pos_Ki[i] += self.vel_Ki[i] * self.delt
                    self.pos_Kp[i] = max(self.Kp_min, min(self.Kp_max, self.pos_Kp[i]))
                    self.pos_Ki[i] = max(self.Ki_min, min(self.Ki_max, self.pos_Ki[i]))

            self.fitness_sum = sum(self.pBest_Fitness)
            self.Mean_Fitness = self.fitness_sum / float(self.NumParticles)
            self.fitness_var = sum((f - self.Mean_Fitness)**2 for f in self.pBest_Fitness)
            self.Fitness_Diversity_Index = self.fitness_var / float(self.NumParticles)

            self.center_Kp = sum(self.pos_Kp) / float(self.NumParticles)
            self.center_Ki = sum(self.pos_Ki) / float(self.NumParticles)
            self.Swarm_Radius = sum(math.sqrt((p - self.center_Kp)**2 + (q - self.center_Ki)**2)
                                    for p, q in zip(self.pos_Kp, self.pos_Ki)) / float(self.NumParticles)

            self.Kp = self.gBest_Kp
            self.Ki = self.gBest_Ki
            self.Global_Fitness = self.gBest_Fitness
            self.Convergence_Cost = abs(self.Mean_Fitness - self.gBest_Fitness)
            self.Iteration_Count += 1
            self.Fitness_Evaluations += int(self.NumParticles * (self.PSO_Iterations + 1))

            self.e_prev = Error
            self.Prev_Actual_Angle = Actual_Angle

        return self.Kp, self.Ki


class Incremental_PI:
    def __init__(self):
        self.e_k1 = 0.0
        self.u_k1 = 0.0
        self.First_Run = True
        self.U_MAX = 220.0   # Aligned with steering saturation
        self.u_k = 0.0
        self.delta_u_k = 0.0

    def reset(self):
        self.e_k1 = 0.0
        self.u_k1 = 0.0
        self.First_Run = True
        self.u_k = 0.0
        self.delta_u_k = 0.0

    def run(self, Enable, e_k, KP, KI):
        if not Enable:
            self.reset()
            return self.u_k, self.delta_u_k

        if self.First_Run:
            self.e_k1 = e_k
            self.u_k1 = 0.0
            self.First_Run = False

        self.delta_u_k = KP * (e_k - self.e_k1) + KI * e_k
        self.delta_u_k = max(-25.0, min(25.0, self.delta_u_k))

        self.u_k = self.u_k1 + self.delta_u_k
        self.u_k = max(-self.U_MAX, min(self.U_MAX, self.u_k))

        self.e_k1 = e_k
        self.u_k1 = self.u_k
        return self.u_k, self.delta_u_k


# =========================================================
#                 MODBUS CONFIGURATION
# =========================================================
PLC_HOST = '192.168.140.1'
PLC_PORT = 502
UNIT = 0x1

client = ModbusClient(PLC_HOST, PLC_PORT)

if not client.connect():
    print("Modbus Connection Failed!")
    exit()
else:
    print("Connected to PLC")

# =========================================================
#                    GLOBAL VARIABLES
# =========================================================
steering_center = 500       # FIXED: Center position for Cytron/PLC

current_angle = 0
current_x = 0
current_y = 0
heading = 0

previous_steer = 0
previous_bearing_diff = 0
previous_steer_output = 0
previous_derivative = 0.0   # For derivative filter
release_counter = 0

launch_mode = True
ramped_cmd = 10
filtered_target_speed = 6

prev_time = None
prev_position = None
current_speed = 0
filtered_speed = 0
previous_cmd = 5

LOG_FILE = "/home/tihan/Downloads/log1.txt"

# =========================================================
# PSO-PI CONTROLLER INSTANCES
# =========================================================
pso_tuner = PSO_Tuner()
incremental_pi = Incremental_PI()

pso_enabled = True
pso_mu_friction = 1.0

# =========================================================
#                   LOGGING FUNCTION
# =========================================================
def log_parameters(current_x, current_y, target_x, target_y,
                   off_x, off_y, heading, bearing, bearing_diff,
                   steer_output, velocity, curvature, wp_switch_dist, cte):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(
            f"Time:{timestamp}, "
            f"Current:({current_x:.3f},{current_y:.3f}), "
            f"Target:({target_x:.3f},{target_y:.3f}), "
            f"Offset:({off_x:.3f},{off_y:.3f}), "
            f"Heading:{heading:.2f}, "
            f"Bearing:{bearing:.2f}, "
            f"BearingDiff:{bearing_diff:.2f}, "
            f"Steer:{steer_output}, "
            f"Velocity:{velocity:.2f}, "
            f"Curvature:{curvature:.5f}, "
            f"SwitchDist:{wp_switch_dist:.2f}, "
            f"CTE:{cte:.3f}\n"
        )

# =========================================================
#              WAYPOINT FILE READER
# =========================================================
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
                if len(parts) < 3:
                    continue
                waypoints.append([float(parts[0]), float(parts[1]), float(parts[2])])
    except Exception as e:
        print("Error reading waypoint file:", e)
    print("Loaded", len(waypoints), "waypoints")
    return waypoints

# =========================================================
#                   PLC FUNCTIONS
# =========================================================
def read_angle():
    global current_angle
    val = client.read_holding_registers(1, 1, unit=UNIT)
    if val:
        current_angle = val.registers[0]
    return current_angle

def set_steer(angle):
    """Map signed angle (-220..+220) to absolute position centered at 500."""
    command = steering_center + int(angle)
    if command > 720:
        command = 720
    elif command < 280:
        command = 280
    client.write_registers(400, command, unit=UNIT)
    client.write_coil(20, angle < 0, unit=UNIT)

def accelerate(value):
    client.write_registers(500, int(value), unit=UNIT)

def control_eff(value):
    try:
        client.write_registers(22, int(value), unit=UNIT)
    except:
        pass

def set_drive_mode(forward=False, reverse=False):
    client.write_coil(502, forward, unit=UNIT)
    client.write_coil(504, reverse, unit=UNIT)

# =========================================================
#          ADAPTIVE STEERING EFFORT
# =========================================================
def set_steering_effort(avg_curvature, cte):
    if abs(cte) > 1.5 or avg_curvature >= 0.02:
        effort = 3300
    elif avg_curvature >= 0.006:
        effort = 2800
    else:
        effort = 2200
    control_eff(effort)

# =========================================================
#                 BRAKE CONTROL
# =========================================================
BRAKE_RELEASE_COIL = 3
BRAKE_APPLY_COIL = 4
last_brake_time = 0
brake_position = 0

def set_brake_coils(c3, c4):
    client.write_coil(BRAKE_RELEASE_COIL, c3, unit=UNIT)
    client.write_coil(BRAKE_APPLY_COIL, c4, unit=UNIT)

def brake_neutral():
    set_brake_coils(False, False)

def apply_brake(level=1, pulse=0.18):
    global last_brake_time, brake_position
    current_time = time.time()
    if current_time - last_brake_time < 0.5:
        return
    brake_position += level
    brake_position = min(brake_position, 5)
    print(f"APPLY BRAKE | Level:{level} | BrakePos:{brake_position}/5")
    for i in range(level):
        set_brake_coils(False, True)
        time.sleep(pulse)
        brake_neutral()
        time.sleep(0.08)
    last_brake_time = current_time

def release_brake_steps(level=1, pulse=0.18):
    global brake_position
    if brake_position <= 0:
        return
    brake_position -= level
    brake_position = max(brake_position, 0)
    print(f"RELEASE BRAKE | Level:{level} | BrakePos:{brake_position}/5")
    for i in range(level):
        set_brake_coils(True, False)
        time.sleep(pulse)
        brake_neutral()
        time.sleep(0.08)

def force_release_brake(level=5, pulse=0.18):
    global brake_position
    print(f"FORCE RELEASING BRAKE ({level} pulses)")
    for i in range(level):
        set_brake_coils(True, False)
        time.sleep(pulse)
        brake_neutral()
        time.sleep(0.08)
    brake_position = 0
    print("Brake Fully Released")

# =========================================================
#              SAFE SHUTDOWN HANDLER
# =========================================================
def shutdown_handler(signum, frame):
    print("\nCTRL+C detected! Stopping vehicle safely...")
    try:
        accelerate(0)
        apply_brake(level=5)
        set_steer(0)
        control_eff(0)
        set_drive_mode(False, False)
        print("Vehicle held for 3 seconds")
        time.sleep(3)
        print("Releasing brake for manual control")
        force_release_brake(level=5)
    except Exception as e:
        print("Error while stopping:", e)
    client.close()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

# =========================================================
#                ROS CALLBACKS
# =========================================================
def callback_ndt_pose(data):
    global current_x, current_y, prev_time, prev_position, current_speed, filtered_speed
    current_x = data.pose.position.x
    current_y = data.pose.position.y
    curr_time = data.header.stamp.to_sec()
    curr_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])

    if prev_time is not None:
        dt = curr_time - prev_time
        if dt > 0:
            displacement = curr_position - prev_position
            velocity = displacement / dt
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2) * 3.6
            current_speed = speed
            alpha_speed = 0.2
            filtered_speed = alpha_speed * speed + (1 - alpha_speed) * filtered_speed

    prev_time = curr_time
    prev_position = curr_position

def callback_eular_angle(data):
    global heading
    heading = data.data[2] * (180 / np.pi)

# =========================================================
#              NEAREST WAYPOINT
# =========================================================
def find_nearest_waypoint_index(waypoints):
    min_distance = float("inf")
    nearest_index = 0
    for i, point in enumerate(waypoints):
        dist = math.hypot(point[0] - current_x, point[1] - current_y)
        if dist < min_distance:
            min_distance = dist
            nearest_index = i
    return nearest_index

# =========================================================
#                BEARING DIFFERENCE
# =========================================================
def fun_bearing_diff(required_bearing):
    bearing_diff = -heading + required_bearing
    if bearing_diff < -180:
        bearing_diff += 360
    if bearing_diff > 180:
        bearing_diff -= 360
    return bearing_diff

# =========================================================
#        FUTURE CURVATURE AVERAGING
# =========================================================
def get_average_future_curvature(waypoints, wp_index):
    future_curvatures = []
    end_index = min(wp_index + 11, len(waypoints))
    for i in range(wp_index, end_index):
        future_curvatures.append(abs(waypoints[i][2]))
    if len(future_curvatures) == 0:
        return 0
    return np.percentile(future_curvatures, 80)

# =========================================================
#          ADAPTIVE LOOKAHEAD DISTANCE
# =========================================================
def get_wp_switch_distance(avg_curvature):
    if avg_curvature < 0.020:
        return 8
    elif avg_curvature < 0.045:
        return 5
    else:
        return 3

# =========================================================
#          CROSS-TRACK ERROR (Stanley)
# =========================================================
def cross_track_error(x, y, waypoints, wp_idx):
    if wp_idx >= len(waypoints) - 1:
        return 0.0
    x1, y1, _ = waypoints[wp_idx]
    x2, y2, _ = waypoints[wp_idx + 1]
    dx = x2 - x1
    dy = y2 - y1
    dvx = x - x1
    dvy = y - y1
    seg_len_sq = dx*dx + dy*dy
    if seg_len_sq < 1e-6:
        return math.hypot(dvx, dvy)
    t = max(0.0, min(1.0, (dvx*dx + dvy*dy) / seg_len_sq))
    cx = x1 + t * dx
    cy = y1 + t * dy
    cross = dx * (y - cy) - dy * (x - cx)
    return math.copysign(math.hypot(x - cx, y - cy), cross)

# =========================================================
#                    ROS SETUP
# =========================================================
def listener():
    rospy.init_node("topic_subscriber")
    rospy.Subscriber("/ndt_pose", PoseStamped, callback_ndt_pose)
    rospy.Subscriber("/eular_angle", Float32MultiArray, callback_eular_angle)

listener()
rospy.wait_for_message('/ndt_pose', PoseStamped)

# =========================================================
#                  LOAD WAYPOINTS
# =========================================================
waypoints = extract_waypoints_from_file("/home/tihan/aroi_avoid/wp/curvature_aml.txt")
wp = find_nearest_waypoint_index(waypoints)

print("Starting WP =", wp)
force_release_brake(level=5)

# =========================================================
#                   CONTROL LOOP
# =========================================================
while not rospy.is_shutdown():

    # ---------------------------------------------------
    # FAILSAFE RELOCALIZATION
    # ---------------------------------------------------
    nearest_wp = find_nearest_waypoint_index(waypoints)
    cte = cross_track_error(current_x, current_y, waypoints, wp)

    if abs(nearest_wp - wp) > 20 and abs(cte) > 3.0:
        wp = nearest_wp
        incremental_pi.reset()  # Reset PI integrator on jump

    # ---------------------------------------------------
    # END CONDITION
    # ---------------------------------------------------
    if wp >= len(waypoints) - 1:
        print("Final waypoint reached")
        accelerate(0)
        apply_brake(level=5)
        set_steer(0)
        set_drive_mode(False, False)
        control_eff(0)
        break

    # ---------------------------------------------------
    # TARGET WAYPOINT
    # ---------------------------------------------------
    target_x, target_y, curvature = waypoints[wp]
    distance = math.hypot(current_x - target_x, current_y - target_y)

    off_x = target_x - current_x
    off_y = target_y - current_y

    # ---------------------------------------------------
    # BEARING
    # ---------------------------------------------------
    bearing = math.degrees(math.atan2(off_y, off_x)) % 360
    bearing_diff = fun_bearing_diff(bearing)

    # ---------------------------------------------------
    # FUTURE CURVATURE
    # ---------------------------------------------------
    avg_curvature = get_average_future_curvature(waypoints, wp)

    # ---------------------------------------------------
    # LOOKAHEAD DISTANCE
    # ---------------------------------------------------
    wp_switch_dist = get_wp_switch_distance(avg_curvature)

    # ---------------------------------------------------
    # ADAPTIVE STEERING EFFORT
    # ---------------------------------------------------
    set_steering_effort(avg_curvature, cte)

    # ---------------------------------------------------
    # TARGET SPEED (FIXED THRESHOLDS: 0.006 / 0.02)
    # ---------------------------------------------------
    if avg_curvature < 0.006:
        target_speed = 10
    elif avg_curvature < 0.02:
        target_speed = 7
    else:
        target_speed = 4

    speed_alpha = 0.08
    filtered_target_speed = (1 - speed_alpha) * filtered_target_speed + speed_alpha * target_speed
    target_speed = filtered_target_speed

    if abs(previous_steer_output) > 120:
        target_speed *= 0.7

    # ---------------------------------------------------
    # BRAKE ASSIST LOGIC (FIXED THRESHOLD: 0.02)
    # ---------------------------------------------------
    brake_active = False
    brake_required = False
    overspeed = filtered_speed - target_speed

    if overspeed > 4 or \
       (avg_curvature > 0.02 and filtered_speed > 7) or \
       (abs(previous_steer_output) > 140 and filtered_speed > 6) or \
       (abs(bearing_diff) > 35 and overspeed > 0.5 and filtered_speed > 5):
        brake_required = True

    if brake_required:
        brake_active = True
        if overspeed > 3 or filtered_speed > 8:
            target_brake_position = 5
        elif overspeed > 2.5:
            target_brake_position = 3
        elif overspeed > 1:
            target_brake_position = 2
        else:
            target_brake_position = 1

        if brake_position < target_brake_position:
            apply_brake(level=(target_brake_position - brake_position))

        ramped_cmd = 0
        cmd = 0
        print(f"BRAKE ASSIST ACTIVE | TargetBrake:{target_brake_position}/5 | CurrentBrake:{brake_position}/5")

    # ---------------------------------------------------
    # SPEED CONTROL
    # ---------------------------------------------------
    error = target_speed - filtered_speed

    if error > 4:
        cmd = 33
    elif error > 2:
        cmd = 30
    elif error > 0.5:
        cmd = 28
    elif error > -0.5:
        cmd = 21
    elif error > -2:
        cmd = 24
    elif error > -4:
        cmd = 13
    else:
        cmd = 10

    cmd = max(15, min(35, cmd))

    if filtered_speed < 2.0:
        launch_mode = True
    else:
        launch_mode = False

    if launch_mode and not brake_active:
        ramped_cmd = 35
        if abs(previous_steer_output) > 20:
            ramped_cmd = 32
    elif not brake_active:
        cmd = 0.4 * previous_cmd + 0.6 * cmd
        previous_cmd = cmd

        if not brake_required and brake_position > 0:
            release_brake_steps(level=1)

        if ramped_cmd < cmd:
            ramped_cmd += 3
        elif ramped_cmd > cmd:
            ramped_cmd -= 3.5

    ramped_cmd = max(0, min(32, ramped_cmd))
    set_drive_mode(forward=True)
    accelerate(int(ramped_cmd))

    # =========================================================
    # PSO-PI STEERING CONTROLLER WITH DAMPING + CTE + SLOPE FF
    # =========================================================

    # Derivative of bearing error (with light filter for noise)
    raw_derivative = bearing_diff - previous_bearing_diff
    derivative = 0.7 * previous_derivative + 0.3 * raw_derivative
    previous_bearing_diff = bearing_diff
    previous_derivative = derivative

    # Run PSO Tuner to get optimal Kp, Ki
    pso_kp, pso_ki = pso_tuner.run(
        Enable=pso_enabled,
        Target_Angle=bearing,
        Actual_Angle=heading,
        Error=bearing_diff,
        U_k=incremental_pi.u_k,
        Mu_friction=pso_mu_friction,
        T_s=0.1
    )

    # Run Incremental PI controller with PSO-tuned gains
    pso_u_k, pso_delta_u = incremental_pi.run(
        Enable=pso_enabled,
        e_k=bearing_diff,
        KP=pso_kp,
        KI=pso_ki
    )

    # Outer-loop derivative damping (higher Kd to kill oscillation)
    if avg_curvature < 0.006:
        Kd_outer = 1.5
        alpha = 0.92
    elif avg_curvature < 0.02:
        Kd_outer = 3.0
        alpha = 0.78
    else:
        Kd_outer = 6.5
        alpha = 0.60

    # Cross-track Stanley correction
    k_cte = 3.0
    steer_cte = math.degrees(math.atan(k_cte * cte / (filtered_speed + 1.0)))

    # Slope feedforward (map-specific)
    slope_ff = 0.0
    if 75.0 <= current_x <= 105.0 and current_y < 0:
        slope_ff = -25.0   # Steep right-side slope: pre-turn left
    elif current_x <= 70.0 and current_y < -10.0:
        slope_ff = 10.0    # Gentle left slope: bias right

    # Combine: PI output + derivative damping + CTE + slope FF
    raw_steer = pso_u_k + Kd_outer * derivative + steer_cte + slope_ff
    raw_steer = max(min(raw_steer, 220), -220)

    # ---------------------------------------------------
    # ADAPTIVE FILTER & RATE LIMIT
    # ---------------------------------------------------
    steep_slope_region = (75.0 <= current_x <= 105.0 and current_y < 0)

    if abs(cte) > 1.5 or steep_slope_region:
        alpha_dyn = 0.50
        max_steer_change = 25
    else:
        alpha_dyn = alpha
        if avg_curvature < 0.006:
            max_steer_change = 8
        elif avg_curvature < 0.02:
            max_steer_change = 12
        else:
            max_steer_change = 25

    smooth_steer = alpha_dyn * previous_steer + (1 - alpha_dyn) * raw_steer
    smooth_steer = max(min(smooth_steer, 220), -220)
    previous_steer = smooth_steer

    delta = smooth_steer - previous_steer_output
    delta = max(min(delta, max_steer_change), -max_steer_change)

    steer_output = previous_steer_output + delta
    steer_output = max(min(steer_output, 220), -220)
    previous_steer_output = steer_output

    # ---------------------------------------------------
    # SEND STEERING
    # ---------------------------------------------------
    set_steer(int(steer_output))

    # ---------------------------------------------------
    # DEBUG PRINT
    # ---------------------------------------------------
    print(
        "WP:", wp,
        "| Dist:", round(distance, 2),
        "| Curv:", round(avg_curvature, 4),
        "| CTE:", round(cte, 2),
        "| BearingDiff:", round(bearing_diff, 2),
        "| Steer:", int(steer_output),
        "| TargetSpeed:", round(target_speed, 1),
        "| CurrentSpeed:", round(filtered_speed, 1),
        "| Overspeed:", round(overspeed, 1),
        "| BrakeReq:", brake_required,
        "| BrakePos:", brake_position,
        f"| CmdTarget:{int(cmd)}",
        f"| RampCmd:{int(ramped_cmd)}",
        f"| PSO_Kp:{pso_kp:.3f}",
        f"| PSO_Ki:{pso_ki:.5f}",
        f"| Kd:{Kd_outer}",
        f"| FF:{slope_ff}",
        f"| Effort:{3300 if (abs(cte)>1.5 or avg_curvature>=0.02) else (2800 if avg_curvature>=0.006 else 2200)}"
    )

    # ---------------------------------------------------
    # LOGGING
    # ---------------------------------------------------
    log_parameters(
        current_x, current_y, target_x, target_y,
        off_x, off_y, heading, bearing, bearing_diff,
        steer_output, filtered_speed, avg_curvature,
        wp_switch_dist, cte
    )

    # ---------------------------------------------------
    # WAYPOINT SWITCHING
    # ---------------------------------------------------
    if distance < wp_switch_dist:
        if wp < len(waypoints) - 1:
            wp += 1

    time.sleep(0.1)