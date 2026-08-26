# .pyc 反编译修复 SOP（Python 3.11）

## 背景
SPUM 项目 `d:\spum-core\经济学\` 下的股票预测模块 .py 源文件全部丢失，仅剩 `__pycache__` 下的 .pyc 编译缓存。
已用 pycdc（Decompyle++）反编译出基础版 .py（存在明显损坏），需要逐文件修复使其可导入且与 .pyc 运行时行为等价。

**已完成的文件**（不要动）：`daily_phase.py`、`industry_mapping.py`、`_stock_minge_v2.py` —— 均已通过等价性验证。

## 关键路径
- 反编译基础版（待修复）：`d:\spum-core\经济学\<模块名>.py`
- 权威 .pyc（运行时真相）：`d:\spum-core\经济学\__pycache__\<模块名>.cpython-311.pyc`
- 辅助工具：`d:\spum-core\tools\pycdc.exe`（反编译，已不用）
- 分析输出目录：`d:\spum-core\tools\`（可写）

## pycdc 对 Python 3.11 的 5 种已知损坏模式（修复重点）

### 模式 1：dataclass 定义损坏
症状：`Xxx = <NODE:12>()`（顶层语法错误）
修复：从 .pyc 反射提取字段定义。命令模板：
```python
import importlib.util, dataclasses, inspect
spec = importlib.util.spec_from_file_location('m', r'd:\spum-core\经济学\__pycache__\<模块名>.cpython-311.pyc')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
f = dataclasses.fields(m.Xxx)
for x in f: print(x.name, x.type, repr(x.default))
# 方法列表（若 dataclass 有自定义方法）
print([n for n in dir(m.Xxx) if not n.startswith('_')])
```
生成形如：
```python
@dataclass
class Xxx:
    a: float = 0.0
    b: str = ''
    # 自定义方法用 dis 还原（见模式 2）
```
注意：dataclass 字段有 default 或 default_factory 之分（用 `x.default is dataclasses.MISSING` 判断）；`field(default_factory=...)` 用 `x.default_factory`。

### 模式 2：函数体丢失
症状：`pass` + `# WARNING: Decompyle incomplete` 或 `return` 被省略
修复：用 dis 还原函数字节码。对能加载的模块直接 dis；对加载失败的模块用 marshal：
```python
import marshal, dis
f = open(r'd:\spum-core\经济学\__pycache__\<模块名>.cpython-311.pyc', 'rb')
f.read(16)
code = marshal.load(f)   # 模块级 code
for c in code.co_consts:
    if hasattr(c, 'co_name') and c.co_name == '<函数名>':
        dis.dis(c)        # 嵌套 code 对象里还有内部函数/闭包
```
dis 输出格式：`行号 OPNAME 参数`。解读要点：
- `LOAD_FAST i (name)` → 读局部变量；`STORE_FAST` → 写
- `LOAD_GLOBAL NULL + name` → 全局名
- `LOAD_CONST` → 常量
- `COMPARE_OP 0 (<) / 5 (>=)` → 比较运算符
- `POP_JUMP_FORWARD_IF_FALSE X (to Y)` → if 分支
- `BINARY_OP 5 (*) / 10 (-) / 11 (/) / 0 (+)` → 算术
- `RETURN_VALUE` → return
- `CALL n` / `PRECALL n` → 函数调用
- `KW_NAMES` → 关键字参数
- `BUILD_CONST_KEY_MAP n` → dict 字面量
- `BUILD_TUPLE n` → 元组；`BUILD_LIST n` → 列表
- `UNPACK_SEQUENCE n` → 解包
- `FOR_ITER` + `JUMP_BACKWARD` → for 循环
- `LOAD_METHOD` → 方法调用
- `COPY / SWAP / STORE_SUBSCR` → `d[k] += 1` 模式
- `LOAD_CLOSURE / MAKE_FUNCTION 8` → 闭包/内部函数（继续 dis 其 code 对象）
- `FORMAT_VALUE` + `BUILD_STRING` → f-string（'火亢单独(前' + format(fire,'.0%') + ')' → f'火亢单独(前{fire:.0%})'）

### 模式 3：变量名丢失成 None
症状：`None.xxx(...)` 或 `... if None else ...`
修复：从上下文判断正确对象名（最常见是 `np`，也可能是局部变量）。用 dis 确认该位置的全局/局部名。

### 模式 4：lambda / walrus / if-else 表达式损坏
症状：`(lambda growth_pct, thresh: g = _positive(growth_pct) / 100if g <= 0.03: ...)()` 等非法语法
修复：通常原始是 `@decorator` 装饰的 def 函数或普通 def。用 dis 还原为普通 def + if/elif/else 语句。注意检查模块顶部是否有 `register` 之类装饰器工厂（该文件 `industry_mapping.py` 已示范：`def register(name): def deco(fn): ...; return deco`）。

### 模式 5：模块级 return
症状：`if __name__ == '__main__':\n    main()\n    return None` —— return 非法
修复：删除孤立 return。原始通常就是 `main()`。

## 等价性验证（完成标准）
对每个修复后的 .py，与 .pyc 用相同输入对比输出。命令模板：
```python
import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
old = load('o', r'd:\spum-core\经济学\__pycache__\<模块名>.cpython-311.pyc')
new = load('n', r'd:\spum-core\经济学\<模块名>.py')
# 对比常量、函数输出、类实例行为（构造代表性输入，含边界值）
```
要求：
1. `py_compile.compile('<模块名>.py', doraise=True)` 语法通过
2. `import` 成功
3. 关键常量/数据相等
4. 每个核心函数在代表性输入（含边界、空、异常值）下输出一致（float 用 abs diff < 1e-9）

## 依赖关系（import 顺序）
- `spum_cycle.py` 依赖 `industry_mapping.py`（已修复完成）
- `spum_backtest.py` 依赖 `spum_cycle` 和 `daily_phase`（均已修复完成）
- 其余模块依赖 pandas/numpy/akshare 等第三方库（环境已装）

## 注意事项
- 禁止删除 .pyc 或 tools/pycdc.exe
- 不要改 .py 中已经正确的部分（docstring、常量表、已恢复的函数）
- 保持函数签名与 .pyc 反射出的 signature 完全一致（用 `inspect.signature` 核对）
- 保持模块公共 API（dir() 可见的非下划线名称）与 .pyc 一致
- 反编译输出顶部的 `# Source Generated with Decompyle++` 注释可保留
- 修复后运行验证脚本，全部通过才算完成
- 完成后在 `d:\spum-core\tools\` 写一个 `<模块名>.done` 标记文件，内容为验证结果摘要
