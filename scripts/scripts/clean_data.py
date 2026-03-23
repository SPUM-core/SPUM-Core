"""
数据清洗脚本：将原始录入的数据统一单位、标准化字段。
"""
import pandas as pd
import numpy as np

def clean_qualitative(input_file, output_file):
    """
    清洗定性数据
    """
    df = pd.read_csv(input_file)
    
    # 单位换算：压强转 Pa
    if 'pressure_pa' not in df.columns and 'pressure_torr' in df.columns:
        df['pressure_pa'] = df['pressure_torr'] * 133.322
    elif 'pressure_pa' not in df.columns and 'pressure_mmhg' in df.columns:
        df['pressure_pa'] = df['pressure_mmhg'] * 133.322
    
    # 温度转 K
    if 'temp_k' not in df.columns and 'temp_c' in df.columns:
        df['temp_k'] = df['temp_c'] + 273.15
    
    # 标准化方向字段
    dir_map = {
        'black_forward': 'black_forward',
        '黑面朝前': 'black_forward',
        'mirror_forward': 'mirror_forward',
        '镜面朝前': 'mirror_forward'
    }
    df['rotation_direction'] = df['rotation_direction'].map(dir_map).fillna(df['rotation_direction'])
    
    # 标准化光照字段
    ill_map = {
        'none': 'none',
        '无光': 'none',
        'diffuse': 'diffuse',
        'direct': 'direct'
    }
    if 'illumination' in df.columns:
        df['illumination'] = df['illumination'].map(ill_map).fillna(df['illumination'])
    
    # 保存
    df.to_csv(output_file, index=False)
    print(f"清洗后数据已保存至 {output_file}")

def clean_quantitative(input_file, output_file):
    """
    清洗定量数据（类似，但保留转速）
    """
    # 与定性类似，增加转速列检查
    df = pd.read_csv(input_file)
    # 单位换算同上...
    # 转速：确保数值类型
    if 'rotation_speed_rpm' in df.columns:
        df['rotation_speed_rpm'] = pd.to_numeric(df['rotation_speed_rpm'], errors='coerce')
    df.to_csv(output_file, index=False)
    print(f"清洗后数据已保存至 {output_file}")

if __name__ == "__main__":
    # 示例用法
    clean_qualitative("data/raw/qualitative_raw.csv", "data/processed/radiometer_rotation_qualitative.csv")
    # clean_quantitative("data/raw/quantitative_raw.csv", "data/processed/radiometer_rotation_quantitative.csv")