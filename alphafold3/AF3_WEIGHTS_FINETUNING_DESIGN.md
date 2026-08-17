# AlphaFold 3 权重接入与微调设计文档

> 状态: 设计已实现（Design → Development → Testing 已闭环）
> 对应实现: [`finetuning/af3/`](../finetuning/af3/)，测试: [`finetuning/tests/`](../finetuning/tests/)
> 上游版本基线: AlphaFold 3 `v3.0.4`（2026-07-28 发布）

---

## 1. 背景：上游权重发布方式已变更

DeepMind 在 2026-07-23 的提交
[`dd1a7bad`](https://github.com/google-deepmind/alphafold3/commit/dd1a7bad)
（*Update the instructions on how to obtain AlphaFold 3 weights*）修改了
`README.md` 与 `docs/installation.md`，取消了此前的申请表流程：

| 项目 | 变更前 | 变更后（当前） |
|------|--------|----------------|
| 权重获取方式 | 填写 Google 表单申请，DeepMind 酌情批准，2–3 个工作日答复 | 直接下载 `https://storage.googleapis.com/alphafold3/af3.bin.zst` |
| 代码许可 | CC BY-NC-SA 4.0 | Apache-2.0（自 `v3.0.3`，2026-06-09 起） |
| 权重许可 | AlphaFold 3 Model Parameters Terms of Use | **未变**（Last Modified 2024-11-09），仍为非商业用途 |
| 运行环境 | 需要 NVIDIA GPU | `v3.0.4` 起支持纯 CPU 与 Apple Silicon |

其它与微调相关的上游事实：

- 参数文件与任意 `3.0.x` 版本兼容；只有发布新模型时才会提升 major/minor 版本号。
- `v3.0.2` 起，上游在 [`docs/model_parameters.md`](https://github.com/google-deepmind/alphafold3/blob/main/docs/model_parameters.md)
  中完整公开了**全部参数的名称、形状与 dtype**。本仓库将该 schema 落地为
  [`finetuning/af3/param_schema.txt`](../finetuning/af3/param_schema.txt)，
  从而可以在**没有真实权重**的情况下构造结构完全一致的随机权重跑通全链路。

### 权重规模（由 schema 推导，非估算）

| 指标 | 数值 |
|------|------|
| 参数条目数 | 405 |
| 参数总量 | 368,384,602（约 368.4 M） |
| `float32` 参数量 | 204,991,834 |
| `bfloat16` 参数量 | 163,392,704 |
| 最大子树 | `diffuser/~/diffusion_head`（约 203.2 M） |
| 次大子树 | `diffuser/evoformer/__layer_stack_no_per_layer_1`（Pairformer 主干 48 层，约 147.4 M） |

---

## 2. 合规约束（对微调的硬性影响）

权重条款（`WEIGHTS_TERMS_OF_USE.md`）中对 “Model Parameters” 的定义包含：

> "(a) modifications to those weights and parameters, (b) works based on those
> weights and parameters, or (c) other code or machine learning models which
> incorporate, in full or in part, those weights and parameters."

由此推导出三条对本仓库微调框架的硬性约束：

1. **微调产物本身也是 Model Parameters。** 全量微调的 checkpoint、合并后的权重，
   以及从 AF3 权重派生的一切结果，都不得公开发布或对组织外共享。
2. **仅非商业主体、非商业用途可用。** 不得用于商业活动，包括代表商业机构开展的研究。
3. **不得用 AF3 的输出训练同类生物分子结构预测模型。**

因此设计上做两件事：

- 明确区分 **LoRA/Adapter 增量参数**（不含原始权重数值，可在组织内自由传递）
  与 **合并后权重**（属于 Model Parameters，受限）。默认只导出前者。
- 在导出接口中内置 `WeightsComplianceError` 守卫：任何会把原始权重数值写入
  产物的操作（合并导出、全量 checkpoint 导出）必须显式传入
  `acknowledge_weights_terms=True`，避免误发布。

> 本仓库**不包含**、也不会分发任何 AlphaFold 3 权重或其派生权重。
> `param_schema.txt` 只含名称/形状/dtype 元数据，来自上游 Apache-2.0 文档。

---

## 3. 目标与非目标

### 目标

- G1 能读写 AF3 官方权重容器格式（`.bin` / `.bin.zst` / 分片），与上游
  `alphafold3.model.params` 二进制兼容。
- G2 能对下载到的权重做结构校验（缺失项、冗余项、形状/dtype 不匹配）。
- G3 能在无真实权重时，按 schema 生成结构一致的随机权重，用于跑通与回归测试。
- G4 能把 405 个扁平参数名映射到功能分组（trunk / diffusion / confidence 等），
  支持“冻结哪些、微调哪些”的策略表达。
- G5 能在 Haiku 参数空间上做 LoRA：为线性层权重挂载低秩增量，支持 layer-stack
  维度，初始增量恒为 0，可合并、可只保存增量。
- G6 提供面向 AF3 的微调入口 `AlphaFold3FineTuner`，串联上述能力，并落实第 2 节合规约束。

### 非目标

- 不实现 AF3 的前向计算与训练 step（上游只发布推理管线；训练代码需自行接入
  JAX，属后续迭代）。本次交付的是**权重层与参数层**基础设施。
- 不下载真实权重进行端到端训练验证（受条款与算力限制）。

---

## 4. 模块设计

```
finetuning/af3/
├── __init__.py          # 对外 API 汇总
├── record_io.py         # G1  官方二进制容器编解码
├── param_schema.txt     # G2/G3 参数 schema（上游 docs 落地）
├── schema.py            # G2/G3 schema 解析、校验、随机权重生成
├── param_groups.py      # G4  参数分组与 LoRA 目标选择
├── lora.py              # G5  Haiku 参数空间 LoRA
├── weights.py           # 下载/加载 CLI 与高层封装
└── finetuner.py         # G6  AlphaFold3FineTuner + 合规守卫
```

### 4.1 `record_io.py`

与上游 `params.py` 的记录格式严格一致：小端 `<5i` 头部
（`len(scope), len(name), len(dtype), len(shape), len(buffer)`），
随后依次是 scope、name、dtype 字符串、`int32` 形状、C 序数组字节。

```python
def encode_record(scope: str, name: str, arr: np.ndarray) -> bytes
def read_records(stream: IO[bytes]) -> Iterator[tuple[str, str, np.ndarray]]
def write_params(path, params: dict[str, dict[str, np.ndarray]], *, compress=None) -> None
def read_params(path_or_dir) -> dict[str, dict[str, np.ndarray]]
def select_model_files(model_dir, model_name=None) -> tuple[list[Path], bool]
```

- `params` 采用与上游一致的两级结构 `{scope: {name: array}}`。
- `compress=None` 时按扩展名推断（`.zst` → zstd）。
- `select_model_files` 复刻上游的 6 条文件名模式（含分片与压缩组合）。
- `zstandard` 为可选依赖，缺失时读写非压缩文件仍可用，遇到 `.zst` 抛
  `MissingDependencyError`。

### 4.2 `schema.py`

```python
@dataclass(frozen=True)
class ParamSpec:
    scope: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    @property
    def full_name(self) -> str      # "scope:name"
    @property
    def num_params(self) -> int

def load_schema(path=None) -> tuple[ParamSpec, ...]
def validate_params(params, schema=None) -> ValidationReport
def generate_random_params(schema=None, *, seed=0, scopes=None) -> dict
def summarize(schema=None) -> SchemaSummary   # 条目数 / 参数量 / dtype 分布
```

`ValidationReport` 字段：`missing`（缺失 full_name）、`unexpected`（多余）、
`shape_mismatch`、`dtype_mismatch`、`ok`（布尔），并提供可读 `describe()`。

`generate_random_params` 对 `__meta__:__identifier__` 填零（与上游文档一致），
其余按 dtype 在 `[-1, 1)` 上均匀采样——避免全零权重被加速器优化掉。

### 4.3 `param_groups.py`

按参数全名归类到 `ParamGroup` 枚举：

| 分组 | 匹配依据 | 约占参数量 |
|------|----------|-----------|
| `EMBEDDING` | `evoformer_conditioning_*`、`*_embed_*`、`left_single`/`right_single` 等输入侧 | ~1.1 M |
| `TEMPLATE` | `template_embedding/*` | ~0.26 M |
| `MSA` | `evoformer/__layer_stack_no_per_layer/`（MSA stack，4 层） | ~3.0 M |
| `PAIRFORMER` | `evoformer/__layer_stack_no_per_layer_1/`（trunk，48 层） | ~147.4 M |
| `DIFFUSION` | `~/diffusion_head/*` | ~203.2 M |
| `CONFIDENCE` | `confidence_head/*` | ~12.9 M |
| `META` | `__meta__:*` | 64 B |

```python
def classify(full_name: str) -> ParamGroup
def group_param_counts(schema=None) -> dict[ParamGroup, int]
def stack_size(full_name, shape) -> int | None   # layer-stack 前导层数，非 stack 返回 None
def is_linear_weight(full_name) -> bool          # ":weights" 且非 layer-norm/scale
def select_lora_targets(schema=None, *, groups=None, patterns=None) -> tuple[str, ...]
DEFAULT_LORA_TARGET_PATTERNS: tuple[str, ...]    # q/k/v/output_projection, transition1/2, gating_query ...
```

`stack_size` 判定：全名含 `__layer_stack_with_per_layer` 或
`__layer_stack_no_per_layer` 时，`shape[0]` 为层数，其余维度才是真实权重形状。

### 4.4 `lora.py`（Haiku 参数空间 LoRA）

对形状为 `(*stack, in_dim, *out_dims)` 的权重，把 `out_dims` 摊平为
`out_dim = prod(out_dims)`，再挂载：

```
lora_a: (*stack, in_dim, rank)   ~ N(0, 1/sqrt(in_dim))
lora_b: (*stack, rank, out_dim)  = 0
delta  = (lora_a @ lora_b) * (alpha / rank) -> reshape 回原始形状
```

```python
@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    seed: int = 0

class AF3LoRA:
    def __init__(self, base_params, target_names, config=LoRAConfig())
    @property
    def targets(self) -> tuple[str, ...]
    def num_lora_params(self) -> int
    def num_base_params(self) -> int
    def delta(self, full_name) -> np.ndarray
    def apply(self) -> dict            # 返回 base + delta 的新参数树（不改原树）
    def state_dict(self) -> dict       # 仅增量，形如 {"full_name": {"lora_a":..,"lora_b":..}}
    def load_state_dict(self, state) -> None
    def save(self, path) -> None       # npz，仅增量
    @classmethod
    def load(cls, path, base_params) -> "AF3LoRA"
```

不变式：`lora_b` 初始为 0 ⇒ `apply()` 在训练前必须与 `base_params` 数值完全一致。

### 4.5 `finetuner.py`

```python
@dataclass
class AF3FineTuneConfig:
    strategy: Literal["lora", "head_only", "full"] = "lora"
    lora: LoRAConfig = LoRAConfig()
    trainable_groups: tuple[ParamGroup, ...] = (ParamGroup.DIFFUSION, ParamGroup.CONFIDENCE)
    lora_groups: tuple[ParamGroup, ...] = (ParamGroup.PAIRFORMER, ParamGroup.DIFFUSION)
    lora_target_patterns: tuple[str, ...] = DEFAULT_LORA_TARGET_PATTERNS

class AlphaFold3FineTuner:
    @classmethod
    def from_pretrained(cls, model_dir, config=None) -> "AlphaFold3FineTuner"
    @classmethod
    def from_random(cls, config=None, *, seed=0) -> "AlphaFold3FineTuner"   # 无权重时的测试入口
    def trainable_param_names(self) -> tuple[str, ...]
    def frozen_param_names(self) -> tuple[str, ...]
    def parameter_summary(self) -> ParameterSummary   # 总量/可训练量/占比/分组明细
    def save_adapter(self, path) -> None              # 仅 LoRA 增量，无条款门槛
    def load_adapter(self, path) -> None
    def export_merged_weights(self, path, *, acknowledge_weights_terms=False) -> None
        # acknowledge_weights_terms 不为 True 时抛 WeightsComplianceError
```

`ParameterSummary.trainable_ratio` 用于回答“LoRA 到底动了多少参数”。

### 4.6 `weights.py`

```python
AF3_WEIGHTS_URL = "https://storage.googleapis.com/alphafold3/af3.bin.zst"
WEIGHTS_TERMS_URL = ".../WEIGHTS_TERMS_OF_USE.md"

def download_weights(dest_dir, *, url=AF3_WEIGHTS_URL, accept_terms=False,
                     expected_sha256=None, chunk_size=1 << 20) -> Path
def load_weights(model_dir) -> dict
def check_weights(model_dir) -> ValidationReport
def main(argv=None) -> int     # python -m finetuning.af3.weights {download,check,info}
```

`accept_terms` 非 True 时抛 `WeightsComplianceError` 并打印条款摘要——
让“我已阅读非商业条款”成为一次显式动作，而不是默默下载。

---

## 5. 测试计划（TDD）

测试位于 `finetuning/tests/`，仅依赖 `numpy` + `pytest`（`zstandard` 可选，缺失则 skip）。

### 5.1 `test_af3_record_io.py`

| # | 用例 | 期望 |
|---|------|------|
| 1 | `encode_record` 头部布局 | 前 20 字节按 `<5i` 解出 5 个长度字段，与 scope/name/dtype/shape/buffer 实际长度一致 |
| 2 | 单条记录往返 | `read_records(encode_record(...))` 得到相同 scope/name/dtype/shape/数值 |
| 3 | 多 dtype 往返 | `float32`/`bfloat16`(以 uint16 视图)/`uint8`/`int32` 逐字节一致 |
| 4 | 参数树往返（非压缩 `.bin`） | `read_params(write_params(p))` 与原树逐数组 `array_equal` |
| 5 | 参数树往返（`.bin.zst`） | 同上；`zstandard` 缺失时 skip |
| 6 | 分片读取 | 写出 `af3.0.bin`/`af3.1.bin` 后 `read_params(dir)` 得到合并结果 |
| 7 | `select_model_files` 模式优先级 | 压缩分片优先于单文件；返回 `is_compressed` 正确 |
| 8 | 截断文件 | 抛 `RecordError` |
| 9 | 空目录 | 抛 `FileNotFoundError` |
| 10 | 与上游格式互认 | 用本模块 `encode_record` 生成的字节，可被本模块按上游头部定义解出（结构断言，见 #1） |

### 5.2 `test_af3_schema.py`

| # | 用例 | 期望 |
|---|------|------|
| 1 | schema 条目数 | 405 |
| 2 | 参数总量 | 368,384,602 |
| 3 | dtype 集合 | `{float32, bfloat16, uint8}` |
| 4 | `__meta__:__identifier__` 存在且 `shape=(64,)`、`dtype=uint8` | 通过 |
| 5 | 关键层存在性 | Pairformer trunk 首维为 48；MSA stack 首维为 4 |
| 6 | `ParamSpec.full_name` / `num_params` | 与手工计算一致 |
| 7 | `generate_random_params` 结构 | 全部 405 项齐备，形状/dtype 与 schema 完全一致 |
| 8 | 随机权重非全零 | `__meta__` 为全零，其余任一大数组不全零 |
| 9 | 随机权重可复现 | 同 seed 两次生成逐字节一致，不同 seed 不同 |
| 10 | `validate_params` 正例 | 随机权重校验 `ok=True`，四类问题列表均为空 |
| 11 | 缺失项 | 删一项 → `missing` 命中该 full_name，`ok=False` |
| 12 | 多余项 | 加一项 → `unexpected` 命中 |
| 13 | 形状不符 | 改形状 → `shape_mismatch` 记录 (期望, 实际) |
| 14 | dtype 不符 | 改 dtype → `dtype_mismatch` 记录 |
| 15 | `summarize` | 条目数/参数量与 #1/#2 一致 |
| 16 | 端到端 | 随机权重 → `write_params` → `read_params` → `validate_params` 仍 `ok` |

### 5.3 `test_af3_param_groups.py`

| # | 用例 | 期望 |
|---|------|------|
| 1 | 每个 schema 条目都能分类 | 无 `UNKNOWN` |
| 2 | 分组归属抽样 | diffusion/pairformer/msa/confidence/template/embedding 代表性名字各归其位 |
| 3 | 分组参数量 | `DIFFUSION` 最大、`PAIRFORMER` 次大；各组之和 == 总参数量 |
| 4 | `stack_size` | `__layer_stack_no_per_layer_1` 的 trunk 权重返回 48；非 stack 参数返回 `None` |
| 5 | `is_linear_weight` | `:weights` 为真；`:scale`、`:bias`、layer-norm 为假 |
| 6 | `select_lora_targets` 默认 | 非空；全部满足 `is_linear_weight`；全部落在默认分组内 |
| 7 | `select_lora_targets(groups=...)` | 限定分组后结果为默认结果的子集 |
| 8 | `patterns` 过滤 | 只给 `q_projection` 时，结果全部含该子串 |

### 5.4 `test_af3_lora.py`

| # | 用例 | 期望 |
|---|------|------|
| 1 | 初始增量为 0 | 所有 target 的 `delta()` 全零 |
| 2 | `apply()` 初始等价 | 与 base 逐数组 `array_equal` |
| 3 | 形状保持 | 每个 target 的 `delta()` 形状 == base 权重形状（含 stack 维） |
| 4 | 非零增量生效 | 手改 `lora_b` 后，`apply()` 仅该 target 变化，其余不变 |
| 5 | 数学正确性（无 stack） | `delta == a @ b * alpha/rank` |
| 6 | 数学正确性（含 stack，48 层） | 逐层 `delta[i] == a[i] @ b[i] * scaling`，多头输出正确 reshape |
| 7 | 参数量 | `num_lora_params == Σ rank*(in+out)*stack`；与 base 之比 < 1% |
| 8 | `state_dict` 只含增量 | 键集合 == targets；不含任何 base 权重数值 |
| 9 | save/load 往返 | 保存后重载，`delta()` 与 `state_dict()` 逐字节一致 |
| 10 | 非法 target | 不存在的名字 → `KeyError`；非线性权重（`:scale`）→ `ValueError` |
| 11 | rank 校验 | `rank <= 0` 或 `rank > min(in,out)` → `ValueError` |
| 12 | 不修改原树 | `apply()` 前后 base 参数树逐数组不变 |

### 5.5 `test_af3_finetuner.py`

| # | 用例 | 期望 |
|---|------|------|
| 1 | `from_random` 构造 | 参数树通过 `validate_params` |
| 2 | LoRA 策略下的可训练集合 | 可训练项 == LoRA 增量项；base 全部冻结 |
| 3 | `head_only` 策略 | 可训练项全部落在 `trainable_groups` 内，且不含 trunk |
| 4 | `full` 策略 | 可训练项 == 全部非 `__meta__` 参数 |
| 5 | `parameter_summary` | LoRA 策略下 `trainable_ratio < 1%`；`full` 策略约为 100% |
| 6 | `save_adapter`/`load_adapter` 往返 | 增量一致；无需条款确认 |
| 7 | adapter 文件不含 base 数值 | 载入 npz 后，任一数组都不等于对应 base 权重 |
| 8 | `export_merged_weights` 缺确认 | 抛 `WeightsComplianceError`，且文件未生成 |
| 9 | `export_merged_weights` 已确认 | 生成文件，`read_params` 后 `validate_params` 仍 `ok` |
| 10 | `download_weights` 缺确认 | 抛 `WeightsComplianceError`，不发起任何网络请求 |
| 11 | CLI `info` | 退出码 0，输出含条目数与参数总量 |
| 12 | 常量正确 | `AF3_WEIGHTS_URL` 指向 `storage.googleapis.com/alphafold3/af3.bin.zst` |

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 未在真实权重上验证 | schema 与上游文档逐条对齐，并提供 `check` 子命令；用户下载后一条命令即可判定加载正确性 |
| 上游发布新模型（3.1.x）导致 schema 变化 | schema 外置为数据文件 + `load_schema(path=...)` 可替换；`validate_params` 会明确报出 missing/unexpected |
| 误发布派生权重 | 合并导出与下载均需显式条款确认；默认导出只含 LoRA 增量 |
| `bfloat16` 在 numpy 中无原生 dtype | 以 `uint16` 视图承载并在 schema 中保留 `bfloat16` 标记，读写逐字节无损 |
| 训练 step 尚未实现 | 明确列为非目标；本次交付权重层/参数层，前向与损失接入作为下一迭代 |

---

## 7. 后续迭代（不在本次范围）

1. 接入 JAX 前向：把 `AF3LoRA.apply()` 的参数树喂给上游 `Diffuser`，实现
   `loss_fn` 与 `optax` 训练循环。
2. `finetuning/heads` 中的任务头与 AF3 `confidence_head` 特征的对接。
3. 蒸馏/输出约束的合规检查（条款禁止用 AF3 输出训练同类结构预测模型）。
