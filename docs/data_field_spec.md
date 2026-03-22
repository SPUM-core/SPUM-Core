# 辐射计数据字段规范

## 数据文件命名
- 定性数据：`radiometer_rotation_qualitative.csv`
- 定量数据：`radiometer_rotation_quantitative.csv`

## 字段说明（定性数据）

| 字段名 | 类型 | 必填 | 说明与示例 |
|--------|------|------|------------|
| `source_id` | string | 是 | 文献唯一标识（DOI、ADS bibcode、或自定义如 “Crookes1874”） |
| `year` | integer | 是 | 出版年份 |
| `author` | string | 否 | 第一作者姓氏 |
| `material_black` | string | 是 | 涂黑面材料（如 “carbon black”, “soot”, “platinum black”） |
| `material_mirror` | string | 是 | 镜面材料（如 “silver”, “aluminum”, “glass”） |
| `gas` | string | 否 | 气体种类，默认 “air” |
| `pressure_pa` | float | 是 | 压强（Pa） |
| `temp_k` | float | 是 | 环境温度（K） |
| `rotation_direction` | string | 是 | 方向：`black_forward` 或 `mirror_forward` |
| `illumination` | string | 是 | 光照条件：`none` / `diffuse` / `direct` |
| `notes` | string | 否 | 备注（如叶片形状、尺寸、实验细节） |

## 字段说明（定量数据，额外字段）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `temp_k` | float | 温度（K） |
| `rotation_speed_rpm` | float | 转速（转/分） |
| `temp_range` | string | 若数据为曲线范围，填写如 “273–373” |

## 单位换算规则
- 压强：1 Torr = 133.322 Pa，1 mmHg = 133.322 Pa
- 温度：℃ → K 加 273.15

## 数据提交方式
将填写好的 CSV 文件放入 `data/processed/` 目录，并在 PR 中说明数据来源。