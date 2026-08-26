# patent —— 专利定位分析目录

> 本目录存放专利申请相关的分析脚本与数据，与临床管线 `五行/人元/脉诊/` 分工：

- **算法文档**（PPG 形态学算法、ECG 技术方案）唯一权威位置：`五行/人元/脉诊/`
- **临床采集管线**：`五行/人元/脉诊/`（采集、处理、五形桥接、CLI）
- **本目录**：专利定位的拟合与分析脚本（PTBXL、自洽拟合、五柱分析等）

> `syndrome_decoder.py` 因被本目录 4 个分析脚本（fitting_detail / comprehensive_fitting / pulse_diagnosis / test_pulse）import，保留副本；它与 `五行/人元/脉诊/syndrome_decoder.py` 保持一致，修改时必须双端同步。

> `datasets/`（butppg / PTBXL / Sleep-EDF 第三方公开数据，约 189 MB）不入库，按 `.gitignore` 排除，本地下载后自行放置。
