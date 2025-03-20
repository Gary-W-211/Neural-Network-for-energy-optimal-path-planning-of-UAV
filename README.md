# Neural-Network-for-energy-optimal-path-planning-of-UAV
数据处理模块 (data_processing.py)：

读取和合并CSV文件

数据清洗和归一化

轨迹拆分

序列数据准备

训练集和测试集划分


神经网络模型 (neural_network_models.py)：

BiLSTM模型实现

TCN模型实现

模型训练函数

能量计算和预测函数

模型评估函数


主函数 (main.py)：

参数解析

数据处理调用

模型训练和评估

模型性能比较


评估函数 (evaluate_model.py)：

轨迹能量预测评估

轨迹特征分析

模型比较

真实数据评估



这个系统可以：

从CSV文件中提取无人机轨迹数据

将数据合并并按3:7的比例分为测试集和训练集

对数据进行归一化处理，尤其是时间数据

使用BiLSTM和TCN两种方法训练神经网络

预测每一个下一时刻的电流和电压

计算能量消耗 (电流 * 电压 * dt)

评估模型性能，包括每个轨迹的RMSE
