import os
import numpy as np
import joblib
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 导入数据处理模块和模型模块
from data_processing import process_drone_data
from neural_network_models import train_and_evaluate, calculate_energy, evaluate_energy_prediction, visualize_results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='无人机轨迹能量预测')
    parser.add_argument('--input_dir', type=str, default='./cleaned_csv', help='输入CSV文件目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出结果目录')
    parser.add_argument('--data_dir', type=str, default='./processed_data', help='处理后的数据目录')
    parser.add_argument('--sequence_length', type=int, default=20, help='序列长度')
    parser.add_argument('--test_size', type=float, default=0.3, help='测试集比例')
    parser.add_argument('--num_splits', type=int, default=5, help='每个轨迹拆分的数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--bilstm_hidden_dim', type=int, default=64, help='BiLSTM隐藏层维度')
    parser.add_argument('--bilstm_num_layers', type=int, default=2, help='BiLSTM层数')
    parser.add_argument('--tcn_channels', type=str, default='32,64,128', help='TCN通道数列表')
    parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN卷积核大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--skip_data_processing', action='store_true', help='跳过数据处理步骤')
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 解析TCN通道数列表
    tcn_channels = [int(c) for c in args.tcn_channels.split(',')]
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 处理数据或加载已处理的数据
    if not args.skip_data_processing:
        print("\n第1步: 处理无人机轨迹数据")
        data = process_drone_data(
            input_dir=args.input_dir,
            output_dir=args.data_dir,
            sequence_length=args.sequence_length,
            test_size=args.test_size,
            num_splits=args.num_splits,
            random_state=args.seed
        )
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        train_trajectories = data['train_trajectories']
        test_trajectories = data['test_trajectories']
        feature_scaler = data['feature_scaler']
        target_scaler = data['target_scaler']
        feature_columns = data['feature_columns']
        target_columns = data['target_columns']
    else:
        print("\n第1步: 加载已处理的数据")
        # 加载已处理的数据
        X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
        train_trajectories = np.load(os.path.join(args.data_dir, 'train_trajectories.npy'))
        test_trajectories = np.load(os.path.join(args.data_dir, 'test_trajectories.npy'))
        feature_scaler = joblib.load(os.path.join(args.data_dir, 'feature_scaler.pkl'))
        target_scaler = joblib.load(os.path.join(args.data_dir, 'target_scaler.pkl'))
        feature_columns = np.load(os.path.join(args.data_dir, 'feature_columns.npy'))
        target_columns = np.load(os.path.join(args.data_dir, 'target_columns.npy'))
    
    # 从训练集中划分出验证集
    X_train, X_val, y_train, y_val, train_trajectories, val_trajectories = train_test_split(
        X_train, y_train, train_trajectories, test_size=0.2, random_state=args.seed
    )
    
    # 打印数据集大小
    print(f"\n训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 训练BiLSTM模型
    print("\n第2步: 训练和评估BiLSTM模型")
    bilstm_model, bilstm_metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        test_trajectories=test_trajectories,
        model_type='bilstm',
        input_dim=X_train.shape[2],  # 特征维度
        hidden_dim=args.bilstm_hidden_dim,
        num_layers=args.bilstm_num_layers,
        output_dim=y_train.shape[1],  # 目标维度 (电流和电压)
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=device,
        output_dir=args.output_dir
    )
    
    # 训练TCN模型
    print("\n第3步: 训练和评估TCN模型")
    tcn_model, tcn_metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        test_trajectories=test_trajectories,
        model_type='tcn',
        input_dim=X_train.shape[2],  # 特征维度
        num_channels=tcn_channels,
        kernel_size=args.tcn_kernel_size,
        output_dim=y_train.shape[1],  # 目标维度 (电流和电压)
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=device,
        output_dir=args.output_dir
    )
    
    # 比较两种模型的性能
    print("\n第4步: 比较BiLSTM和TCN模型性能")
    
    # 创建比较图表
    plt.figure(figsize=(12, 8))
    
    # 比较RMSE
    metrics_comparison = {
        'BiLSTM': bilstm_metrics['rmse'],
        'TCN': tcn_metrics['rmse']
    }
    
    plt.subplot(2, 2, 1)
    plt.bar(metrics_comparison.keys(), metrics_comparison.values())
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 比较MAE
    metrics_comparison = {
        'BiLSTM': bilstm_metrics['mae'],
        'TCN': tcn_metrics['mae']
    }
    
    plt.subplot(2, 2, 2)
    plt.bar(metrics_comparison.keys(), metrics_comparison.values())
    plt.title('MAE Comparison')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 比较R²
    metrics_comparison = {
        'BiLSTM': bilstm_metrics['r2'],
        'TCN': tcn_metrics['r2']
    }
    
    plt.subplot(2, 2, 3)
    plt.bar(metrics_comparison.keys(), metrics_comparison.values())
    plt.title('R² Comparison')
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存比较结果
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'))
    
    print("\n模型比较结果:")
    print(f"BiLSTM RMSE: {bilstm_metrics['rmse']:.6f}")
    print(f"TCN RMSE: {tcn_metrics['rmse']:.6f}")
    print(f"BiLSTM MAE: {bilstm_metrics['mae']:.6f}")
    print(f"TCN MAE: {tcn_metrics['mae']:.6f}")
    print(f"BiLSTM R²: {bilstm_metrics['r2']:.6f}")
    print(f"TCN R²: {tcn_metrics['r2']:.6f}")
    
    # 确定性能最佳的模型
    if bilstm_metrics['rmse'] < tcn_metrics['rmse']:
        best_model = 'BiLSTM'
        best_metrics = bilstm_metrics
    else:
        best_model = 'TCN'
        best_metrics = tcn_metrics
    
    print(f"\n在RMSE指标上，{best_model}模型性能更好")
    
    # 保存比较结果
    comparison_results = {
        'bilstm_metrics': bilstm_metrics,
        'tcn_metrics': tcn_metrics,
        'best_model': best_model
    }
    
    joblib.dump(comparison_results, os.path.join(args.output_dir, 'model_comparison_results.pkl'))
    
    print(f"\n完成! 所有结果已保存到 {args.output_dir}")
    
if __name__ == "__main__":
    main()