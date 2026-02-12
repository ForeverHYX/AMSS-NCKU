import sys
import os
import numpy as np

def load_dat(filepath):
    """加载 .dat 文件，跳过注释行。"""
    return np.loadtxt(filepath, comments="#")

def calculate_asc26_rms(baseline_file, optimized_file):
    """
    完全对齐 ASC26 任务书公式的 RMS 计算。
    公式: RMS = sqrt( (1/M) * sum( ((r1 - r2) / max(r1, r2))^2 ) )
    """
    if not os.path.exists(baseline_file) or not os.path.exists(optimized_file):
        print(f"  错误: 找不到轨迹文件。")
        return None

    data_base = load_dat(baseline_file)
    data_opt = load_dat(optimized_file)

    # 取两者共同的时间步长 M
    M = min(len(data_base), len(data_opt))
    if M == 0:
        return None

    # 提取所有坐标列: BH1_x,y,z 和 BH2_x,y,z (列索引 1 到 6)
    # data_base 形状为 (M, 7)，取后 6 列
    r_base = data_base[:M, 1:7]
    r_opt = data_opt[:M, 1:7]

    # 1. 计算分子: |r1i - r2i|
    numerator = np.abs(r_base - r_opt)

    # 2. 计算分母: max(r1, r2)
    # 按照任务书截图语义，分母取两组数据的对应坐标最大值。
    # 为了防止分母为 0 导致崩溃，且考虑到物理坐标的对称性，
    # 这里使用 np.maximum(np.abs(r_base), np.abs(r_opt)) 是最稳健的。
    denom = np.maximum(np.abs(r_base), np.abs(r_opt))

    # 3. 掩码处理：避免除以零
    mask = denom > 1e-12

    # 4. 计算相对误差的平方
    # 注意：这里是将所有坐标分量（6个）合在一起计算
    rel_error_sq = np.zeros_like(r_base)
    rel_error_sq[mask] = (numerator[mask] / denom[mask]) ** 2

    # 5. 计算 RMS
    # 按照任务书公式，对所有 M 步进行求和并除以 M。
    # 在 6 维情况下，mean() 等同于对所有维度求和后除以 (M * 6)
    total_rms = np.sqrt(np.mean(rel_error_sq))
    
    return total_rms

def check_constraint(optimized_output_dir):
    """检查 ADM 约束是否满足绝对值 < 2。"""
    path = os.path.join(optimized_output_dir, "bssn_constraint.dat")
    if not os.path.exists(path):
        return False
    
    data = load_dat(path)
    # 提取 Level 0 数据 (假定网格层级为 9)
    level0_data = data[::9] 
    # 检查 Ham, Px, Py, Pz (列索引 1-4)
    max_violation = np.max(np.abs(level0_data[:, 1:5]))
    return max_violation

def main():
    if len(sys.argv) < 3:
        print("用法: python3 final_check.py <基准目录> <优化目录>")
        sys.exit(1)

    base_dir = sys.argv[1]
    opt_dir = sys.argv[2]

    base_bh = os.path.join(base_dir, "AMSS_NCKU_output/bssn_BH.dat")
    opt_bh = os.path.join(opt_dir, "AMSS_NCKU_output/bssn_BH.dat")
    opt_out_dir = os.path.join(opt_dir, "AMSS_NCKU_output")

    print("=" * 60)
    print("ASC26 AMSS-NCKU 结果一致性终极验证")
    print("=" * 60)

    # 1. RMS 验证
    rms = calculate_asc26_rms(base_bh, opt_bh)
    if rms is not None:
        status = "PASS" if rms < 0.01 else "FAIL"
        print(f"[1] 轨迹 RMS 误差 (需 < 1%): {rms*100:.6f}%  [{status}]")
    else:
        print("[1] 轨迹 RMS 误差: 失败 (找不到数据)")

    # 2. 约束验证
    max_c = check_constraint(opt_out_dir)
    if max_c is not False:
        status = "PASS" if max_c < 2.0 else "FAIL"
        print(f"[2] ADM 约束最大值 (需 < 2.0): {max_c:.6f}  [{status}]")
    else:
        print("[2] ADM 约束验证: 失败 (找不到数据)")

    # 3. 文件检查
    fig_path = os.path.join(opt_dir, "figure/BH_Trajectory_XY.pdf")
    file_status = "PASS" if os.path.exists(fig_path) else "FAIL"
    print(f"[3] 必备 PDF 图像文件检查: [{file_status}]")

    print("-" * 60)

if __name__ == "__main__":
    main()