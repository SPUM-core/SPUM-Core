import numpy as np

# 定义引力常数（适配SPARC数据单位：kpc, km/s, M⊙）
G = 4.30091e-6  # (km/s)² * kpc / M⊙

def load_and_process_data(filepath):
    """
    加载并处理SPARC星系数据文件（如UGC128.dat）
    参数：
        filepath: 数据文件路径（如 "sparc_data/UGC128.dat"）
    返回：
        r: 半径 (kpc)
        v_obs: 观测旋转速度 (km/s)
        err_v: 速度误差 (km/s)
        L_cum: 累积3.6μm光度 (L⊙)
        Mgas_cum: 累积气体质量 (M⊙)
    """
    # 读取数据：增强容错，跳过注释行和空行
    try:
        data = np.genfromtxt(
            filepath, 
            comments='#', 
            skip_header=0,
            invalid_raise=False,
            filling_values=0.0
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到数据文件：{filepath}，请检查路径是否正确（参考路径：sparc_data/UGC128.dat）")
    except Exception as e:
        raise ValueError(f"读取文件失败：{e}")
    
    # 移除全NaN行，确保数据是二维数组
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data = data[~np.all(np.isnan(data), axis=1)]
    
    # 校验列数（SPARC数据至少5列）
    if data.shape[1] < 5:
        raise ValueError(f"文件 {filepath} 列数不足5列（SPARC标准），实际{data.shape[1]}列")
    
    # 提取基础列
    r = data[:, 0]       # 半径 (kpc)
    v_obs = data[:, 1]   # 观测速度 (km/s)
    err_v = data[:, 2]   # 速度误差 (km/s)
    I_36 = data[:, 3]    # 3.6μm面光度 (L⊙/pc²)
    v_gas = data[:, 4]   # 气体速度 (km/s)
    
    # 数据清洗：过滤无效值（半径/速度需>0，无无穷值）
    mask = (
        (r > 0) & 
        (v_obs > 0) & 
        (err_v > 0) & 
        np.isfinite(I_36) & 
        np.isfinite(v_gas)
    )
    r = r[mask]
    v_obs = v_obs[mask]
    err_v = err_v[mask]
    I_36 = I_36[mask]
    v_gas = v_gas[mask]
    
    # 校验有效数据点数量
    if len(r) < 3:
        raise ValueError(f"有效数据点不足3个（仅{len(r)}个），请检查数据文件是否完整")
    
    # 计算累积光度（关键修复：kpc→pc单位转换）
    dr = np.gradient(r)                # 半径间隔 (kpc)
    r_pc = r * 1000                    # 转换为秒差距 (pc)
    dr_pc = dr * 1000                  # 半径间隔转换为pc
    dL = I_36 * (2 * np.pi * r_pc * dr_pc)  # 环带光度 (L⊙)
    L_cum = np.cumsum(dL)              # 累积光度 (L⊙)
    
    # 计算累积气体质量（修复逻辑：区分面密度和速度估算）
    Mgas_cum = np.zeros_like(r)
    if data.shape[1] >= 6:
        gas_col = data[mask, 5]
        # 若第6列是气体质量面密度（M⊙/pc²），则积分计算累积质量
        if np.all(np.isfinite(gas_col)) and np.max(gas_col) > 0:
            dMgas = gas_col * (2 * np.pi * r_pc * dr_pc)  # 环带气体质量 (M⊙)
            Mgas_cum = np.cumsum(dMgas)
            print("提示：使用第6列气体质量面密度计算累积气体质量")
        else:
            # 否则用气体速度动力学估算
            Mgas_cum = (v_gas**2 * r) / G
            print("提示：第6列无效，用气体速度估算累积气体质量")
    else:
        # 无第6列时，动力学估算
        Mgas_cum = (v_gas**2 * r) / G
    
    # 限制异常值（防止数值溢出）
    Mgas_cum = np.minimum(Mgas_cum, 1e11)
    
    return r, v_obs, err_v, L_cum, Mgas_cum

# 测试调用（核心：匹配你的真实数据文件路径）
if __name__ == "__main__":
    # 替换为你的实际路径（参考解码结果）
    data_path = "sparc_data/UGC128.dat"  
    try:
        r, v_obs, err_v, L_cum, Mgas_cum = load_and_process_data(data_path)
        print("✅ 数据处理成功！")
        print(f"📊 有效数据点：{len(r)} 个")
        print(f"🔍 半径范围：{np.min(r):.2f} ~ {np.max(r):.2f} kpc")
        print(f"💡 累积光度范围：{np.min(L_cum):.2e} ~ {np.max(L_cum):.2e} L⊙")
        print(f"🌌 累积气体质量范围：{np.min(Mgas_cum):.2e} ~ {np.max(Mgas_cum):.2e} M⊙")
    except Exception as e:
        print(f"❌ 运行失败：{e}")
