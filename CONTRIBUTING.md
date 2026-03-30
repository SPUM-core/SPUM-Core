# open_SPUM 贡献指南
感谢您对SPUM理论的关注与贡献！本指南将帮助您完成从草稿提交到核心库升级的全流程。

## 一、贡献范围
我们接纳以下类型的贡献：
1. **理论迭代**：SPUM核心理论的完善、修正、扩展
2. **实验方案**：四项核心预言的实验设计、验证方案、数据记录
3. **模拟代码**：关系网络演化、拓扑动力学、光子尾迹等模拟实现
4. **文档优化**：理论文档的通俗化、可视化、翻译、勘误
5. **工具开发**：评分脚本、验证工具、可视化平台的开发

## 二、提交流程
### 1. 提交草稿
1. **Fork仓库**：点击Gitee页面右上角「Fork」，将仓库克隆到您的个人账号
2. **创建草稿目录**：在`draft/`下创建目录，格式为`draft/[你的主题]/[版本号]/`
   - 主题命名：简洁清晰，如`temperature_torque_experiment`、`consciousness_vspt_extension`
   - 版本号：从v0.1开始迭代
3. **编写内容**：
   - 理论/实验文档：命名为`theory.md`/`experiment.md`，使用Markdown格式
   - 初始化元数据：创建`version.json`，内容如下（版本号留空，由CI自动生成）
     ```json
     {
       "version": "",
       "content_type": "theory",
       "files": ["theory.md"],
       "merkle_root": "",
       "timestamp": 0,
       "score": {
         "A": 0,
         "B": 0,
         "C": 0,
         "D": 0,
         "LE": 0
       }
     }
4.提交代码：在您的 Fork 仓库中提交修改，提交信息格式为[draft] 主题: 简要说明
5.发起 PR：向本仓库master分支发起 Pull Request，标题格式为[草稿提交] 主题名称