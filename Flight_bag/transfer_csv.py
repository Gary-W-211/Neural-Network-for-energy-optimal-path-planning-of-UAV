#!/usr/bin/env python3
import os
import rosbag
import csv

def extract_fields_from_msg(topic, msg, t_stamp):

    row = {
        "time": t_stamp.to_sec(),
        # telemetry111 
        "rate_roll": None, "rate_pitch": None, "rate_yaw": None,
        "current": None, "voltage": None,
        # estimator111 
        "pos_x": None, "pos_y": None, "pos_z": None,
        "vel_x": None, "vel_y": None, "vel_z": None,
        "roll": None, "pitch": None, "yaw": None
    }
    
    if topic == "/telemetry111":
        # rategyro
        try:
            
            row["rate_roll"] = msg.rateGyro[0]
            row["rate_pitch"] = msg.rateGyro[1]
            row["rate_yaw"] = msg.rateGyro[2]
        except Exception as e:
            print(f"Error reading rateGyro in /telemetry111: {e}")
        # current
        try:
            row["current"] = msg.current
        except AttributeError as e:
            print(f"Error reading current in /telemetry111: {e}")
        # voltage
        try:
            row["voltage"] = msg.voltage
        except AttributeError as e:
            print(f"Error reading voltage in /telemetry111: {e}")
    
    elif topic == "/estimator111":
        # position
        try:
            row["pos_x"] = msg.posx
            row["pos_y"] = msg.posy
            row["pos_z"] = msg.posz
        except AttributeError as e:
            print(f"Error reading position in /estimator111: {e}")
        # velocity
        try:
            row["vel_x"] = msg.velx
            row["vel_y"] = msg.vely
            row["vel_z"] = msg.velz
        except AttributeError as e:
            print(f"Error reading velocity in /estimator111: {e}")
        #orientation
        try:
            row["roll"] = msg.attroll
            row["pitch"] = msg.attpitch
            row["yaw"] = msg.attyaw
        except AttributeError as e:
            print(f"Error reading orientation in /estimator111: {e}")
    
    return row

def bag_to_csv(bag_path, output_dir):
    bag = rosbag.Bag(bag_path, 'r')
    bag_basename = os.path.splitext(os.path.basename(bag_path))[0]
    csv_filename = f"{bag_basename}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    print(f"正在将 {bag_path} 转换为 CSV 文件：{csv_path} ...")
    
    header = [
        "time",
        "rate_roll", "rate_pitch", "rate_yaw",
        "current", "voltage",
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "roll", "pitch", "yaw"
    ]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for topic, msg, t_stamp in bag.read_messages():
            if topic not in ["/telemetry111", "/estimator111"]:
                continue
            row = extract_fields_from_msg(topic, msg, t_stamp)
            writer.writerow(row)
    bag.close()
    print(f"CSV 文件已生成：{csv_path}")

def process_folder(rosbag_path, csv_path):
    for filename in os.listdir(rosbag_path):
        if filename.endswith(".bag"):
            bag_path = os.path.join(rosbag_path, filename)
            print(f"正在处理文件：{bag_path} ...")
            bag_to_csv(bag_path, csv_path)
    print("所有 rosbag 文件转换完成。")

if __name__ == "__main__":
    # 修改为你的 rosbag 文件夹和 CSV 输出文件夹路径
    rosbag_path = "./Flight_bag"
    csv_path = "./Flight_csv"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    process_folder(rosbag_path, csv_path)
