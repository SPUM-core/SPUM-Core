"""青檬引擎 · 接口层 —— 对外入口

对外提供 CLI 调试工具（`python -m qingmeng.api.cli`）与统一引擎入口
（`qingmeng.QingmengEngine`）。REST API（FastAPI）为可选依赖，
见 pyproject `api` extra。

注意：本模块不顶层导入 cli（避免 `-m` 执行时的模块重入警告），
CLI 请直接以 `python -m qingmeng.api.cli` 运行。
"""
