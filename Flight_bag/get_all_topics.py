#!/usr/bin/env python3
import os
import rosbag
import csv

def extract_msg_fields(msg):
    """
    通用地提取消息中的所有字段。通过检查 __slots__ 属性，遍历所有字段。
    """
    data = {}
    if hasattr(msg, '__slots__'):
        for slot in msg.__slots__:
            try:
                value = getattr(msg, slot)
                # 如果属性本身也有 __slots__（嵌套消息），递归提取并以 parent.child 的形式记录
                if hasattr(value, '__slots__'):
                    nested_data = extract_msg_fields(value)
                    for k, v in nested_data.items():
                        data[f"{slot}.{k}"] = v
                else:
                    data[slot] = value
            except Exception as e:
                data[slot] = f"Error: {e}"
    else:
        data['data'] = str(msg)
    return data

def bag_to_csv(bag_path, output_dir):
    bag = rosbag.Bag(bag_path, 'r')
    bag_basename = os.path.splitext(os.path.basename(bag_path))[0]
    # 获取所有话题
    topics = list(bag.get_type_and_topic_info()[1].keys())
    
    for topic in topics:
        # 用下划线替换斜杠，生成文件名，比如 "bagName_drone_odom.csv"
        topic_filename = topic.replace("/", "_")
        csv_filename = f"{bag_basename}{topic_filename}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        print(f"Writing CSV for topic {topic} to {csv_path}...")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = None
            header_written = False
            
            # 正确解包返回值：topic, msg, t_stamp
            for _topic, msg, t_stamp in bag.read_messages(topics=[topic]):
                msg_data = extract_msg_fields(msg)
                # 添加时间字段
                try:
                    msg_data["time"] = t_stamp.to_sec()
                except Exception as e:
                    msg_data["time"] = f"Error: {e}"
                # 写入 CSV 表头
                if not header_written:
                    header = list(msg_data.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=header)
                    writer.writeheader()
                    header_written = True
                writer.writerow(msg_data)
    bag.close()

def process_folder(directory, output_dir):
    for filename in os.listdir(directory):
        if filename.endswith(".bag"):
            bag_path = os.path.join(directory, filename)
            print(f"Processing {bag_path}...")
            bag_to_csv(bag_path, output_dir)
    print("所有 rosbag 文件转换完成。")

if __name__ == "__main__":
    rosbag_dir = "./Flight_bag"   # 请修改为你的 rosbag 文件夹路径
    csv_output_dir = "./Flight_csv"   # CSV 文件输出目录
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
    process_folder(rosbag_dir, csv_output_dir)
