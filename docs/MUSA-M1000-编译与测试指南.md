# 在 MTT M1000 (mp_22) 上编译并运行 MUSA 后端的 SGLang

本文档记录在摩尔线程 **MTT M1000**（架构 `mp_22`，aarch64，MUSA SDK 5.1.0）上从源码编译 SGLang MUSA 后端、并加载 Qwen3-0.6B 完成推理验证的完整流程。

预编译 wheel（`torch_musa`/`sglang-kernel` 官方发布版）面向 **S5000 (mp_31) + x86_64 + cp310**，与 M1000 不兼容，因此所有组件均需本机源码编译。

## 1. 环境要求

| 组件 | 要求 | 本机情况 |
|---|---|---|
| GPU | MTT M1000（`mp_22`，warp=128 线程） | ✓ |
| MUSA SDK | ≥ 5.1.0（`/usr/local/musa`，`mcc` 可用） | 5.1.0 |
| Python | 3.10（venv） | /usr/bin/python3.10 |
| 网络 | 可访问 `dl.mthreads.com` 与 PyPI（间歇 DNS 抖动需重试） | ✓ |

> 注：系统若缺 `python3.10-venv`/`python3.10-dev` 且无 sudo，用 `python3.10 -m venv --without-pip` + `get-pip.py`，Python.h 从 timeshift 备份拷贝（见 §3 兼容层）。

## 2. 创建 venv 与安装 torch_musa 栈

```bash
cd /home/roy/gitrepo/github/sglang
python3.10 -m venv --without-pip venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
./venv/bin/python /tmp/get-pip.py
./venv/bin/pip install -U pip 'setuptools<82' wheel

# 摩尔线程 PyPI 源
MT=https://dl.mthreads.com/repo/api/pypi/pypi/simple

# aarch64 + cp310 可用的 mp_22 wheel（torch 与 torch_musa 必须同版本）
./venv/bin/pip install --no-deps "torch==2.11.0.post1+musa5.2.1mp22" --index-url $MT
./venv/bin/pip install --no-deps "torch_musa==2.11.0.post1+musa5.2.1mp22" --index-url $MT
./venv/bin/pip install --no-deps torchada==0.1.82          # PyPI（避免拉入 CUDA torch）
./venv/bin/pip install --no-deps "torchvision==0.26.0.post1+musa5.2.1mp22" --index-url $MT
./venv/bin/pip install --no-deps triton==3.2.0 --index-url $MT
./venv/bin/pip install typing_extensions filelock sympy networkx jinja2 fsspec "numpy<2" ninja
```

## 3. SDK 5.1 vs wheel 5.2 兼容层（一次性）

torch wheel 按 musa5.2.1 构建，本机 SDK 是 5.1.0，有两处符号/soname 缺口，用用户目录补齐（无需 root）：

```bash
mkdir -p ~/.local/musa-compat
ln -sf /usr/local/musa/lib/libmupti.so.1.2 ~/.local/musa-compat/libmupti.so.1   # soname libmupti.so.1

# Python.h（系统无 python3.10-dev 时，从 timeshift 备份恢复）
mkdir -p ~/.local/musa-compat/include_python310/aarch64-linux-gnu/python3.10
cp /backup/timeshift/snapshots/<snapshot>/localhost/usr/include/python3.10/*.h \
   ~/.local/musa-compat/include_python310/
cp /backup/timeshift/snapshots/<snapshot>/localhost/usr/include/aarch64-linux-gnu/python3.10/pyconfig.h \
   ~/.local/musa-compat/include_python310/aarch64-linux-gnu/python3.10/
```

运行任何 Python 前需设置（建议写进 `venv/bin/activate`）：

```bash
export LD_PRELOAD=/usr/local/musa/lib/libmusparse.so                    # 补 libmusa_kernels.so 的 musparse 符号
export LD_LIBRARY_PATH=$HOME/.local/musa-compat:/usr/local/musa/lib
export CPATH=$HOME/.local/musa-compat/include_python310                 # triton JIT 编 driver.c 用
```

验证：

```bash
./venv/bin/python -c "
import torch, torch_musa
print(torch.musa.is_available())   # True
print(torch.musa.get_device_properties(0).major, torch.musa.get_device_properties(0).minor)  # 2 2 → mp_22
"
```

## 4. 编译 sglang-kernel（AOT，csrc/musa）

MUSA 专用内核源码在 `python/sglang/kernels/aot/csrc/musa/`（`.mu` 文件），由 `setup_musa.py` 用 `mcc` 编译（本机 mcc 会经 ssh 调远程编译服务，属正常现象，耗时长）。

```bash
cd python/sglang/kernels/aot
cp pyproject.toml pyproject_cuda_backup.toml
cp pyproject_musa.toml pyproject.toml

CPATH=$HOME/.local/musa-compat/include_python310 \
MTGPU_TARGET=mp_22 \
SGLANG_MUSA_THIRD_PARTY_DIR=$PWD/build/_deps \
LD_PRELOAD=/usr/local/musa/lib/libmusparse.so \
LD_LIBRARY_PATH=$HOME/.local/musa-compat:/usr/local/musa/lib \
/home/roy/gitrepo/github/sglang/venv/bin/python setup_musa.py install
```

说明：
- 首次构建自动克隆 mutlass / flashinfer 到 `build/_deps`。
- `MTGPU_TARGET=mp_22` 对应 M1000；S5000 用 `mp_31`。
- **mp_22 关键源码修改**：`csrc/musa/moe_gemv_swiglu.mu` 中 FP8 模板使用了 `__musa_e4m32f16_rn_bst4` intrinsic（仅 mp_31 存在），本分支已加 `#ifndef ENABLE_FP8` 守卫——不定义 `ENABLE_FP8` 时跳过 FP8 实例化，运行时遇到 FP8 输入会报清晰错误（原逻辑本就拒绝 arch<300 的 FP8）。

验证：

```bash
./venv/bin/python -c "
import torch, torch_musa, sgl_kernel
from sgl_kernel.musa import top_k_top_p_sampling_from_probs
p  = torch.rand(4, 100, device='musa')
tk = torch.full((4,1), 5, dtype=torch.int32, device='musa')
tp = torch.full((4,1), 0.9, device='musa')
print(top_k_top_p_sampling_from_probs(p, tk, tp).shape)  # (4,)
"
```

## 5. 安装 sglang 主包

```bash
cd /home/roy/gitrepo/github/sglang/python
cp pyproject.toml pyproject_cuda_backup.toml
cp pyproject_other.toml pyproject.toml
```

`pyproject_other.toml` 的 `srt_musa` 依赖是为 S5000 CI 内部源写的（deep_ep/tilelang_musa/mthreads-ml-py 等在公开源缺 aarch64 wheel，mate 0.2.7 全家桶又要求 tvm-ffi≥0.1.11 而本机只有 0.1.9.post3）。M1000 不需要 mate（见 §6 注意事项），本分支已将 `srt_musa` 精简为公开源可解的集合。

由于 pip 解析器在弱网下会回溯到远古 sdist，采用 **`--no-deps` + 手动补依赖** 的稳妥路径：

```bash
MT=https://dl.mthreads.com/repo/api/pypi/pypi/simple
V=/home/roy/gitrepo/github/sglang/venv/bin/pip

# 1) sglang 本体（editable，方便二次开发）
$V install -e . --no-deps --no-build-isolation

# 2) runtime 依赖
$V install "numpy<2" "pillow>=11" "transformers==5.12.1" "openai==2.6.1" \
  fastapi uvicorn uvloop pyzmq pydantic orjson msgspec einops gguf interegular \
  "llguidance>=1.7.6,<2.0.0" "mistral_common>=1.11.5" modelscope partial_json_parser \
  prometheus-client psutil py-spy pybase64 python-multipart scipy sentencepiece \
  "soundfile==0.13.1" tiktoken xxhash easydict datasets compressed-tensors \
  "outlines==0.1.11" "timm==1.0.16" "xgrammar" "openai-harmony==0.0.4" \
  "smg-grpc-servicer>=0.9.0" --extra-index-url $MT

# 3) 若 datasets 装完报 fsspec 冲突
$V install "fsspec[http]<=2026.6.0"
```

**mp_22 源码修改（主包）**：`python/sglang/srt/layers/moe/topk.py` 原在 MUSA 下 `import mate` 失败即抛 ImportError；本分支改为 `moe_fused_gate = None` 并守卫调用分支，缺失时回退纯 PyTorch topk。

## 6. 加载 Qwen3-0.6B 测试

M1000 注意事项：
- `attention_backend` 必须用 **`triton`**（MUSA fa3 后端要求 MP≥31 即 S5000）。
- `disable_cuda_graph=True`（M1000 上 graph 捕获 50 个 shape 极慢，单 shape 约 2 分钟）。
- `mem_fraction_static≈0.7`（过高曾触发 host OOM）。
- mate / flash_attn_3 / deep-gemm 0.2.7 全家桶未安装（无 aarch64 wheel 且面向 S5000），走 triton/muDNN 路径；FP8 模型不可用（M1000 硬件不支持）。

```python
# /tmp/test_musa_e2e.py
import torch, torch_musa
import sglang as sgl

if __name__ == "__main__":
    print("musa avail:", torch.musa.is_available())
    llm = sgl.Engine(
        model_path="/home/roy/gitrepo/models/Qwen3-0.6B",
        attention_backend="triton",
        mem_fraction_static=0.7,
        disable_cuda_graph=True,
    )
    out = llm.generate(
        ["The capital of China is"],
        sampling_params={"max_new_tokens": 16, "temperature": 0},
    )
    print("OUTPUT:", out[0]["text"])
    llm.shutdown()
    print("E2E OK")
```

```bash
cd /home/roy/gitrepo/github/sglang
MTHREADS_VISIBLE_DEVICES=0 ./venv/bin/python /tmp/test_musa_e2e.py
```

实测输出：

```
musa avail: True
OUTPUT:  Beijing. The capital of the United States is Washington, D.C. The capital
E2E OK
```

## 7. 本分支改动清单（musa-m1000-support）

| 文件 | 改动 | 原因 |
|---|---|---|
| `python/sglang/kernels/aot/csrc/musa/moe_gemv_swiglu.mu` | FP8 模板加 `#ifndef ENABLE_FP8` 守卫 | `__musa_e4m32f16_rn_bst4` intrinsic 仅 mp_31 有 |
| `python/sglang/srt/layers/moe/topk.py` | mate 缺失时回退纯 PyTorch topk | M1000 无 mate 栈 |
| `python/pyproject.toml`、`python/sglang/kernels/aot/pyproject.toml` | 换用 MUSA 配置并精简 `srt_musa` 依赖 | 公开源 aarch64 可解 |

**真正的源码级修复只有前两行（且仅 mp_22 需要）。** 注意区分：

### 工作区里 ~100 个文件的"通用代码改动"是构建工具链的临时改写，不是本分支的功能改动

编译后 `git status` 会看到大量 `csrc/**`、`include/**`、`jit/include/**` 被改（`ATen/cuda→torch_musa/csrc/aten/musa` include 替换、`CUDAGuard→MUSAGuard`、PTX 内联 asm 加 `if(0)` 守卫等）。这些改动的文件 mtime 全部等于编译启动时刻，是 `torch_musa` 构建系统在编译前的原地改写（musify 类机制），**不是需要提交的代码**。判断依据：

- 上游 main 分支的同一文件（如 `csrc/allreduce/custom_all_reduce.cu`）至今仍是 `#include <ATen/cuda/Exceptions.h>`；
- S5000 的 CI（`scripts/ci/musa/musa_install_dependency.sh`）没有任何源码补丁步骤，只做 `mv pyproject_musa.toml pyproject.toml && setup_musa.py install`——S5000 用户装预编译 wheel，更看不到这些 diff；
- 重新编译时工具链会再次自动改写这些文件。

因此提交时应只保留上表 4 处改动 + 本文档；工具链改写留在工作区即可（或加入 `.gitignore`/构建前 stash）。

## 8. S5000 上 sglang 不改通用代码就跑 MUSA 的机制

| 层 | 机制 | 说明 |
|---|---|---|
| C++/内核编译期 | `torch_musa` 的 `MUSAExtension`/`BuildExtension`（`torch_musa/utils/musa_extension.py`）+ musify 工具（`/usr/local/musa/bin/musify-text`，Aho-Corasick 标识符替换） | 编译前原地改写源码：CUDA 头/API → torch_musa 对应物，`.cu` 由 `mcc -x musa -mtgpu` 编译。改写只发生在构建环境，wheel 编好后用户无感知 |
| 头文件解析 | `torch_musa/share/generated_cuda_compatible/`（编译命令 `-I` 注入） | 提供整套 ATen/c10 头的 MUSA 版本，未改写的 `#include <ATen/...>` 也能解析到 MUSA 实现 |
| Python 运行时 | sglang 上游显式的 `is_musa()` 分支 + `sglang/srt/hardware_backend/musa/` 专属实现（attention backend、topk kernels 等） | 本来就是上游维护的设备无关抽象，为 MUSA 预写，无需"改" |
| Python 生态 | torchada 适配层 + dl.mthreads.com PyPI 源的 `+musa` wheel（mate、flash_attn_3、deep-gemm 等） | S5000 有完整的 mate 加速栈；M1000/mp_22 无 aarch64 wheel 且 mubin 面向 S5000，故走 triton/muDNN 路径 |

所以 M1000 与 S5000 的差异只在：mp_22 缺 FP8 intrinsic（需 `moe_gemv_swiglu.mu` 宏守卫）、无 mate 栈（需 topk 回退）、fa3 后端要求 MP≥31（用 `--attention-backend triton`）。
