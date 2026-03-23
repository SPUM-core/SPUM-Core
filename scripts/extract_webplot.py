"""
从图表图像中提取数据点的辅助工具（使用 webplotdigitizer API 或手动）。
此脚本仅提供调用框架，实际使用时需用户手动操作 WebPlotDigitizer。
"""
import pandas as pd

def manual_entry_from_plot():
    """
    手动输入从图表读取的数据点。
    """
    # 示例：用户从图表读取一系列 (temp, speed) 点
    data = [
        (280, 12),
        (290, 8),
        (300, 0),
        (310, -5),  # 负号表示方向反转
        (320, -10)
    ]
    df = pd.DataFrame(data, columns=['temp_k', 'rotation_speed_rpm'])
    df['rotation_direction'] = df['rotation_speed_rpm'].apply(lambda x: 'black_forward' if x > 0 else ('mirror_forward' if x < 0 else 'none'))
    return df

if __name__ == "__main__":
    df = manual_entry_from_plot()
    df.to_csv("data/processed/extracted_plot_data.csv", index=False)