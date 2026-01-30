from collections import defaultdict
import csv
import pandas as pd
import numpy as np

import time

#
# log mode setting
# 

LOG_NOTHING = 0
LOG_PERFORMANCE = 1
LOG_COLLISION_INFO = 2

log_mode = LOG_NOTHING

def is_log_nothing():
    return log_mode == LOG_NOTHING

def is_log_collision():
    return log_mode == LOG_COLLISION_INFO

#
# reocrd frame count setting
#
record_frame_count = 100



##############
# scene dict #
##############

scene_dict = defaultdict(list)
# v- num_vert 
# v- num_edge 
# v- num_face 
# v- time_step 
# v- subiter_per_step 
# v- mu (friction_coefficient) collosion info지만 씬에 대해 고정이라 일단 여기.
def record_to_scene(key, value):
    if is_log_nothing():
        return 
    if not scene_dict.get(key):
        scene_dict[key] = value
        print("scene info, ", key, ': ', value)


def export_scene_dict_to_csv(filepath="scene_data.csv"):
    """Export scene_dict to a CSV file.
    
    Args:
        filepath: Path to the output CSV file.
    """
    if not scene_dict:
        print("scene_dict is empty. Nothing to export.")
        return
    
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header and values
            for key, value in scene_dict.items():
                writer.writerow([key, value])
        
        print(f"scene_dict exported to {filepath}")
    except IOError as e:
        print(f"Error writing to file: {e}")



##############
# frame dict #
##############

record_frame_finished = False
export_needed = False
frame_log_list = [
    # "frame_idx", 
    # "total_time",
    # "broad_time",
    # "narrow_time",
    # # "col_detect_time",
    # "body_cloth_col_count",
    # "max_penetration_depth",
]

frame_dict = defaultdict(list)
# v- frame_idx: 프레임 인덱스 0부터 시작.  
# v- total_time: 한 프레임 당 계산 시간 
# v- broad_time: 한 프레임 내 broad phase 계산 시간 
# v- narrow_time: 한 프레임 내 narrow phase 계산 시간 
# v- col_detect_time: 한 프레임 내 collision detection 계산에 쓴 시간 
# v- body_cloth_col_count: 한 프레임 내 감지된 body-cloth 충돌 pair 수 
# - max_penetration_depth: 모든 충돌 중 가장 큰 penetration depth 수치

def export_frame_dict_to_csv(filepath="frame_data.csv"):
    """Export frame_dict to a CSV file.
    
    Args:
        filepath: Path to the output CSV file.
    """
    if not frame_dict or not frame_dict["frame_idx"]:
        print("frame_dict is empty. Nothing to export.")
        return
    
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            headers = list(frame_dict.keys())
            writer.writerow(headers)
            
            # Write rows
            num_rows = len(frame_dict["frame_idx"])
            for i in range(num_rows):
                row = [frame_dict[key][i] for key in headers]
                writer.writerow(row)
        
        print(f"frame_dict exported to {filepath}")
    except IOError as e:
        print(f"Error writing to file: {e}")


##################
# iteration dict #
##################

iteration_dict = defaultdict(list)
# v- frame_idx: 프레임 인덱스 
# v- substep_idx: 한 스텝 계산 위한 솔버의 sub_step 수 
# v- iteration_idx: 솔버 한 sub_step에서 반복하는 iteration 인덱스. 
# v- force_residual: 솔버가 줄여야하는 force의 residual 
# v- cloth_self_vt_col_count: 
# v- cloth_self_ee_col_count: 

def export_iteration_dict_to_csv(filepath="iteration_data.csv"):
    """Export iteration_dict to a CSV file.
    
    Args:
        filepath: Path to the output CSV file.
    """
    if not iteration_dict or not iteration_dict.get("frame_idx"):
        print("iteration_dict is empty. Nothing to export.")
        return
    
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            headers = list(iteration_dict.keys())
            writer.writerow(headers)
            
            # Write rows
            num_rows = len(iteration_dict["frame_idx"])
            for i in range(num_rows):
                row = [iteration_dict[key][i] for key in headers]
                writer.writerow(row)
        
        print(f"iteration_dict exported to {filepath}")
    except IOError as e:
        print(f"Error writing to file: {e}")


##################
# collision dict #
##################

collision_dict = defaultdict(list)
# v- contact_idx: 충돌 idx 
# v- frame_idx: 충돌이 발생한 프레임 인덱스 
# v- is_self_col: self collision인가?
# v- is_vt_col: 점-면 충돌인가? (self-collision의 경우,)
# v- is_ee_col: 선-선 충돌인가? (self-collision의 경우,)
# - is_body_cloth_col: 강체-옷 충돌인가?
# - vertex_id: 점-면 충돌일 경우 점 인덱스
# - face_id: 점-면 충돌일 경우 면 인덱스
# - edge_id1: 선-선 충돌일 경우 선 인덱스
# - edge_id2: 선-선 충돌일 경우 선 인덱스
# v- t_x, t_y, t_z (T)
# v- b_x, b_y, b_z (T)
# v- contact_normal_x, contact_normal_y, contact_normal_z (collision_normal)
# v- u_norm
# v- eps_u (slip threshold)
# v- `is_slip` = (u_norm > eps_u)
# v- `force_x,force_y,force_z` (마찰 force)
# v- `normal_contact_force` (N)
def export_collision_dict_to_parquet(filepath="collision_data.parquet"):
    """Export collision_dict to a Parquet file.
    
    Args:
        filepath: Path to the output Parquet file.
    """
    if not collision_dict or "contact_idx" not in collision_dict or not collision_dict["contact_idx"]:
        print("collision_dict is empty. Nothing to export.")
        return
    
    try:
        print("Creating df...")
        df = pd.DataFrame(collision_dict)
        print("Created!")
        
        if df.empty:
            print("⚠ No collisions found. Skipped saving.")
            return
        
        print("Let's save!")
        df.to_parquet(filepath, index=False)
        print(f"✅ Exported {len(df)} rows to {filepath}")
        
    except Exception as e:
        print(f"❌ Error exporting parquet: {e}")

def export_collision_dict_to_csv(filepath="collision_data.csv"):
    """Export collision_dict to a CSV file.
    
    Args:
        filepath: Path to the output CSV file.
    """
    if not collision_dict or "contact_idx" not in collision_dict or not collision_dict["contact_idx"]:
        print("collision_dict is empty. Nothing to export.")
        return
    
    try:
        print("Creating df...")
        df = pd.DataFrame(collision_dict)
        print("Created!")
        
        if df.empty:
            print("⚠ No collisions found. Skipped saving.")
            return
        
        print("Let's save!")
        df.to_csv(filepath, index=False)
        print(f"✅ Exported {len(df)} rows to {filepath}")
        
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")


########################
# collision_per_v dict #
########################

collision_per_v_dict = defaultdict(list)
def export_collision_per_v_dict_to_csv(filepath="collision_per_v_data.csv"):
    """Export collision_per_v_dict to a CSV file.
    
    Args:
        filepath: Path to the output CSV file.
    """
    if not collision_per_v_dict or "contact_idx" not in collision_per_v_dict or not collision_per_v_dict["contact_idx"]:
        print("collision_per_v_dict is empty. Nothing to export.")
        return
    
    try:
        print("Creating df...")
        df = pd.DataFrame(collision_per_v_dict)
        print("Created!")
        
        if df.empty:
            print("⚠ No collision per-vertex data found. Skipped saving.")
            return
        
        print("Let's save!")
        df.to_csv(filepath, index=False)
        print(f"✅ Exported {len(df)} rows to {filepath}")
        
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")


def export_collision_per_v_dict_to_parquet(filepath="collision_per_v_data.parquet"):
    """Export collision_per_v_dict to a Parquet file.
    
    Args:
        filepath: Path to the output Parquet file.
    """
    if not collision_per_v_dict or "contact_idx" not in collision_per_v_dict or not collision_per_v_dict["contact_idx"]:
        print("collision_per_v_dict is empty. Nothing to export.")
        return
    
    try:
        print("Creating df...")
        df = pd.DataFrame(collision_per_v_dict)
        print("Created!")
        
        if df.empty:
            print("⚠ No collision per-vertex data found. Skipped saving.")
            return
        
        print("Let's save!")
        df.to_parquet(filepath, index=False)
        print(f"✅ Exported {len(df)} rows to {filepath}")
        
    except Exception as e:
        print(f"❌ Error exporting parquet: {e}")


### Util Functions!!!!!!!! ###

def frame_idx():
    if len(frame_dict["frame_idx"]):
        return frame_dict["frame_idx"][-1]
    else:
        return -1

def frame_start(frame):
    if is_log_nothing():
        return 

    global record_frame_finished
    global export_needed

    if frame > 0 and frame % 10 == 0:
        export_needed = True

    if export_needed:
        export_scene_dict_to_csv("scene_data.csv")
        if log_mode == LOG_PERFORMANCE:
            export_frame_dict_to_csv("frame_data.csv")
            export_iteration_dict_to_csv("iteration_data.csv")
        elif log_mode == LOG_COLLISION_INFO:
            # export_collision_dict_to_csv("collision_data.csv")
            export_collision_dict_to_parquet(f"collision_data_{frame}.parquet")
            # export_collision_per_v_dict_to_csv("collision_per_v_data.csv")
            export_collision_per_v_dict_to_parquet(f"collision_per_v_data_{frame}.parquet")
            collision_dict.clear()
            collision_per_v_dict.clear()
        else:
            print("Not Logging")
        # export_collision_dict_to_csv("collision_data.csv")
        export_needed = False

    if record_frame_finished:
        return

    num_row = len(frame_dict["frame_idx"])
    for key in frame_dict.keys():
        if len(frame_dict[key]) != num_row:
            print("Warning! frame_dict is not stacked properly.")
            record_frame_finished = True
            return

    if frame < record_frame_count:
        frame_dict["frame_idx"].append(frame)
        print("------ frame", frame_dict["frame_idx"][-1], " --------")
    else:
        record_frame_finished = True
        export_needed = True

def record_to_frame(key, value):
    if is_log_nothing():
        return 
        
    if record_frame_finished:
        return
    frame = frame_idx()
    if frame < 0:
        print("Try to append '", key, "' but frame is not started. Skipped.")
        return

    if len(frame_dict[key]) < frame+1:
        frame_dict[key].append(value)
    elif len(frame_dict[key]) == frame+1:
        frame_dict[key][frame] += value
    else:
        print("Late key insertion tried: ", key, ". Skipped.")

    if key in frame_log_list:
        print("frame[", frame, "]info, ", key, ': ', frame_dict[key][frame])

iteration_log_list = [
    # "force_residual",
    # "cloth_self_vt_col_count",
    # "cloth_self_ee_col_count",
]
def substep_start(substep):
    if is_log_nothing():
        return 

    global record_frame_finished

    frame = frame_idx()
    if frame < 0:
        print("Try to append 'substep' but frame is not started. Skipped.")
        return
    
    if record_frame_finished:
        return

    num_row = len(iteration_dict["frame_idx"])
    for key in iteration_dict.keys():
        if len(iteration_dict[key]) != num_row:
            print("Warning! iteration_dict is not stacked properly: ", key)
            record_frame_finished = True
            return

    # add two first
    iteration_dict["substep_idx"].append(substep)
    if len(iteration_dict["frame_idx"]) < len(iteration_dict["substep_idx"]):
        iteration_dict["frame_idx"].append(frame)

def get_substep():
    return iteration_dict["substep_idx"][-1]


def record_to_iteration(key, value, iter_num):
    if is_log_nothing():
        return 

    if record_frame_finished:
        return

    frame = frame_idx()
    if frame < 0:
        print("Try to append '", key, "' but frame is not started. Skipped.")
        return
    if iter_num < -1:
        return

    # add all (frame and substep are added if needed)
    if len(iteration_dict["iteration_idx"]) == 0 or iteration_dict["iteration_idx"][-1] != iter_num: # should add new row
        if len(iteration_dict["frame_idx"]) == len(iteration_dict["iteration_idx"]):
            iteration_dict["frame_idx"].append(frame)
            iteration_dict["substep_idx"].append(iteration_dict["substep_idx"][-1])
        iteration_dict["iteration_idx"].append(iter_num)
        iteration_dict["force_residual"].append(0)
        iteration_dict["cloth_self_vt_col_count"].append(0)
        iteration_dict["cloth_self_ee_col_count"].append(0)
    iteration_dict[key][-1] = value
    
    if key in iteration_log_list:
        print("  substep[", iteration_dict["substep_idx"][-1], "], iter[", iteration_dict["iteration_idx"][-1], "]info, ", key, ': ', iteration_dict[key][-1])



def record_to_collision_per_v(frame, substep, iter_num, contact_idx,
                                vert_id,
                                is_vert, is_facet, is_edge,
                                normal_contact_force_x, normal_contact_force_y, normal_contact_force_z,
                                friction_x, friction_y, friction_z, mu):
    collision_per_v_dict["frame_idx"].append(frame)
    collision_per_v_dict["substep_idx"].append(substep)
    collision_per_v_dict["iter_num"].append(iter_num)
    collision_per_v_dict["contact_idx"].append(contact_idx)

    collision_per_v_dict["vert_id"].append(vert_id)
    collision_per_v_dict["is_vert"].append(is_vert)
    collision_per_v_dict["is_facet"].append(is_facet)
    collision_per_v_dict["is_edge"].append(is_edge)
    collision_per_v_dict["normal_contact_force_x"].append(normal_contact_force_x)
    collision_per_v_dict["normal_contact_force_y"].append(normal_contact_force_y)
    collision_per_v_dict["normal_contact_force_z"].append(normal_contact_force_z)
    collision_per_v_dict["friction_x"].append(friction_x)
    collision_per_v_dict["friction_y"].append(friction_y)
    collision_per_v_dict["friction_z"].append(friction_z)
    collision_per_v_dict["friction_mu"].append(mu)





def record_to_collision(iter_num,
                        contact_idx,
                        is_self_col, is_body_cloth_col,
                        is_vt_col, is_ee_col,
                        vertex_id, facet_id, edge_id1, edge_id2,
                        Tx, Ty, Tz, Bx, By, Bz, Nx, Ny, Nz, # tangent, bitanget, normal frame
                        u_norm, eps_u,
                        is_slip,
                        normal_force_sum_x, normal_force_sum_y, normal_force_sum_z,
                        normal_force_min_x, normal_force_min_y, normal_force_min_z,
                        friction_x, friction_y, friction_z,
                        v_list,
                        normal_contact_force0_x,
                        normal_contact_force0_y,
                        normal_contact_force0_z,
                        normal_contact_force1_x,
                        normal_contact_force1_y,
                        normal_contact_force1_z,
                        normal_contact_force2_x,
                        normal_contact_force2_y,
                        normal_contact_force2_z,
                        normal_contact_force3_x,
                        normal_contact_force3_y,
                        normal_contact_force3_z,
                        friction0_x,
                        friction0_y,
                        friction0_z,
                        friction1_x,
                        friction1_y,
                        friction1_z,
                        friction2_x,
                        friction2_y,
                        friction2_z,
                        friction3_x,
                        friction3_y,
                        friction3_z,
                        mu
                        ):
    if is_log_nothing():
        return 
    global record_frame_finished

    if record_frame_finished:
        return

    frame = frame_idx()
    if frame < 0:
        print("Try to append collision data but frame is not started. Skipped.")
        return
    
    substep = get_substep()
    
    for i in range(len(contact_idx)):
        if friction_x[i] == 0 and friction_y[i] == 0 and friction_z[i] == 0: 
            continue # skip not important data

        # Append iteration number
        collision_dict["frame_idx"].append(frame)
        collision_dict["substep_idx"].append(substep)
        collision_dict["iter_num"].append(iter_num)

        # Append scalar and vector data
        collision_dict["contact_idx"].append(contact_idx[i])
        collision_dict["is_self_col"].append(is_self_col[i])
        collision_dict["is_body_cloth_col"].append(is_body_cloth_col[i])
        collision_dict["is_vt_col"].append(is_vt_col[i])
        collision_dict["is_ee_col"].append(is_ee_col[i])

        collision_dict["vertex_id"].append(vertex_id[i])
        collision_dict["facet_id"].append(facet_id[i])
        collision_dict["edge_id1"].append(edge_id1[i])
        collision_dict["edge_id2"].append(edge_id2[i])





        # Tangent, bitangent, normal vectors
        collision_dict["Tx"].append(Tx[i])
        collision_dict["Ty"].append(Ty[i])
        collision_dict["Tz"].append(Tz[i])
        collision_dict["Bx"].append(Bx[i])
        collision_dict["By"].append(By[i])
        collision_dict["Bz"].append(Bz[i])
        collision_dict["Nx"].append(Nx[i])
        collision_dict["Ny"].append(Ny[i])
        collision_dict["Nz"].append(Nz[i])
        
        # Contact parameters
        collision_dict["u_norm"].append(u_norm[i])
        collision_dict["eps_u"].append(eps_u[i])
        collision_dict["is_slip"].append(is_slip[i])
        collision_dict["normal_force_sum_x"].append(normal_force_sum_x[i])
        collision_dict["normal_force_sum_y"].append(normal_force_sum_y[i])
        collision_dict["normal_force_sum_z"].append(normal_force_sum_z[i])
        collision_dict["normal_force_min_x"].append(normal_force_min_x[i])
        collision_dict["normal_force_min_y"].append(normal_force_min_y[i])
        collision_dict["normal_force_min_z"].append(normal_force_min_z[i])
        collision_dict["friction_x"].append(friction_x[i])
        collision_dict["friction_y"].append(friction_y[i])
        collision_dict["friction_z"].append(friction_z[i])
        collision_dict["friction_mu"].append(mu[i])

        if is_vt_col[i]: # 4 friction data
            record_to_collision_per_v(frame, substep, iter_num, contact_idx[i],
                                    v_list[i][0],
                                    False, #is_vert, 
                                    True, #is_facet, 
                                    False, #is_edge,
                                    normal_contact_force0_x[i], normal_contact_force0_y[i], normal_contact_force0_z[i],
                                    friction0_x[i], friction0_y[i], friction0_z[i], mu[i])
            record_to_collision_per_v(frame, substep, iter_num, contact_idx[i],
                                    v_list[i][1],
                                    False, #is_vert, 
                                    True, #is_facet, 
                                    False, #is_edge,
                                    normal_contact_force1_x[i], normal_contact_force1_y[i], normal_contact_force1_z[i],
                                    friction1_x[i], friction1_y[i], friction1_z[i], mu[i])
            record_to_collision_per_v(frame, substep, iter_num, contact_idx[i],
                                    v_list[i][2],
                                    False, #is_vert, 
                                    True, #is_facet, 
                                    False, #is_edge,
                                    normal_contact_force2_x[i], normal_contact_force2_y[i], normal_contact_force2_z[i],
                                    friction2_x[i], friction2_y[i], friction2_z[i], mu[i])
            record_to_collision_per_v(frame, substep, iter_num, contact_idx[i],
                                    v_list[i][3],
                                    True, #is_vert, 
                                    False, #is_facet, 
                                    False, #is_edge,
                                    normal_contact_force3_x[i], normal_contact_force3_y[i], normal_contact_force3_z[i],
                                    friction3_x[i], friction3_y[i], friction3_z[i], mu[i])
        elif is_ee_col[i]:
            record_to_collision_per_v(frame, substep, iter_num, contact_idx[i],
                                    v_list[i][0],
                                    False, #is_vert, 
                                    False, #is_facet, 
                                    True, #is_edge,
                                    normal_contact_force0_x[i], normal_contact_force0_y[i], normal_contact_force0_z[i],
                                    friction0_x[i], friction0_y[i], friction0_z[i], mu[i])
            record_to_collision_per_v(frame, substep, iter_num, contact_idx[i],
                                    v_list[i][1],
                                    False, #is_vert, 
                                    False, #is_facet, 
                                    True, #is_edge,
                                    normal_contact_force1_x[i], normal_contact_force1_y[i], normal_contact_force1_z[i],
                                    friction1_x[i], friction1_y[i], friction1_z[i], mu[i])
        elif is_body_cloth_col[i]:
            record_to_collision_per_v(frame, substep, iter_num, contact_idx[i],
                                    v_list[i][0],
                                    True, #is_vert, 
                                    False, #is_facet, 
                                    False, #is_edge,
                                    normal_contact_force0_x[i], normal_contact_force0_y[i], normal_contact_force0_z[i],
                                    friction0_x[i], friction0_y[i], friction0_z[i], mu[i])
        else:
            print("IDK what collision is, Stop recording")
            record_frame_finished = True


    # for key in collision_dict.keys():
    #     print(f"{key}: {len(collision_dict[key])}")
    # print(len(collision_dict["contact_idx"]))

    num_row = len(collision_dict["frame_idx"])
    for key in collision_dict.keys():
        if len(collision_dict[key]) != num_row:
            print("Warning! collision_dict is not stacked properly: ", key)
            record_frame_finished = True
            return



##### Stopwatch!!!!!!!!!!! #####


class Stopwatch:
    def __init__(self):
        self.acc_time = 0
        self.start_time = 0
    
    def clear(self):
        self.acc_time = 0

    def start(self):
        self.start_time = time.perf_counter()
    
    def clear_start(self):
        self.clear()
        self.start()

    def stop(self):
        self.acc_time += time.perf_counter() - self.start_time



frame_timer = Stopwatch()
broad_timer = Stopwatch()
narrow_timer = Stopwatch()

