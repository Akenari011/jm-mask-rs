# jm-mask-rs

二次元 NSFW 打码服务的 **Rust 实现**，用于替代 Python 版 `onnx_api.py`（FastAPI + onnxruntime）。

接口与 Python 版**完全兼容**，MaiBot 插件 `麦麦看本` 无需任何改动。

## 为什么用它

| | Python 版 (onnx_api.py) | **Rust 版 (jm-mask-rs)** |
|---|---|---|
| 部署物 | 729 MB venv + Python 版本对齐 | **1.5 MB 单文件二进制** |
| 依赖 | fastapi/uvicorn/numpy/opencv/onnxruntime | 仅系统 `libonnxruntime.so` |
| 常驻内存 | 492 MB | **411 MB**（60 次请求后仍稳定，无泄漏）|
| 单张延迟 | ~230 ms | **~190–215 ms** |
| 冷启动 | 数秒 | 毫秒级 |

> 内存没降到几十 MB 的原因：真正占内存的是 **onnxruntime 加载模型 + 1024×1024 推理的中间张量**，不是 Python 运行时。
> 本项目的价值主要在**部署简单**（单二进制）与**启动快**，其次才是内存。


## 预编译二进制（推荐）

到 [Releases](https://github.com/Akenari011/jm-mask-rs/releases) 下载对应平台，**无需安装 Rust**：

| 平台 | 文件 | 运行时还需要 |
|---|---|---|
| Linux x86_64 | `jm-mask-rs-linux-x86_64` | `libonnxruntime.so` + 两个 `.onnx` 模型 |
| Linux aarch64 | `jm-mask-rs-linux-aarch64` | `libonnxruntime.so` + 两个 `.onnx` 模型 |
| Windows x86_64 | `jm-mask-rs-windows-x86_64.exe` | `onnxruntime.dll` + 两个 `.onnx` 模型 |
| macOS Apple Silicon | `jm-mask-rs-macos-arm64.tar.gz` | 包内 `libonnxruntime.dylib` + 两个 `.onnx` 模型 |
| macOS Intel | `jm-mask-rs-macos-x86_64.tar.gz` | 包内 `libonnxruntime.dylib` + 两个 `.onnx` 模型 |

校验：同目录 `SHA256SUMS.txt`。

> **关于“自包含”**：二进制本身不静态链接 onnxruntime（`ldd` 里看不到它），
> 运行时通过 `MASK_ORT_DYLIB` 动态加载，因此 Linux/Windows 仍需单独提供
> `libonnxruntime.so` / `onnxruntime.dll`。完整可移植部署一共 4 个文件：
> 二进制 + 运行时库 + 分割模型 + 分类模型（约 104 MB），不需要 Python。

### 最小可移植部署（Linux）

```bash
mkdir -p jm-mask && cd jm-mask
# 放入 jm-mask-rs、libonnxruntime.so、segmentation_model.onnx、classification_model.onnx
MASK_ADDR=127.0.0.1:8000 \
MASK_ORT_DYLIB="$PWD/libonnxruntime.so" \
MASK_SEG_MODEL="$PWD/segmentation_model.onnx" \
MASK_CLS_MODEL="$PWD/classification_model.onnx" \
./jm-mask-rs
```

### macOS 说明

macOS **没有静态链接版**：onnxruntime 官方 macOS 预编译库内含 CoreML 的 Objective-C++ 对象，
静态链接会缺 `___isPlatformVersionAtLeast` 符号（`ld: symbol(s) not found for architecture arm64`）。
因此 macOS 采用**动态加载 + 随包附带 dylib**：

```bash
tar xzf jm-mask-rs-macos-arm64.tar.gz && cd jm-mask-rs-macos-arm64
xattr -dr com.apple.quarantine .   # 首次运行如被 Gatekeeper 拦
export MASK_ORT_DYLIB="$PWD/libonnxruntime.1.22.0.dylib"
export MASK_SEG_MODEL=/path/to/segmentation_model.onnx
export MASK_CLS_MODEL=/path/to/classification_model.onnx
./jm-mask-rs
```

## 从源码构建

```bash
# 默认：静态链接 onnxruntime（自包含，约 25 MB）
cargo build --release

# 可选：动态加载系统 libonnxruntime.so（体积仅 1.5 MB，运行时需该库）
cargo build --release --no-default-features --features dynamic-ort
```

## 运行

```bash
MASK_ADDR=127.0.0.1:8000 \
MASK_SEG_MODEL=/path/to/segmentation_model.onnx \
MASK_CLS_MODEL=/path/to/classification_model.onnx \
MASK_ORT_DYLIB=/path/to/libonnxruntime.so.1.32.0 \
./target/release/jm-mask-rs
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MASK_ADDR` | `127.0.0.1:8000` | 监听地址（**只绑回环**） |
| `MASK_SEG_MODEL` | `models/segmentation_model.onnx` | 分割（检测头）模型 |
| `MASK_CLS_MODEL` | `models/classification_model.onnx` | 分类模型 |
| `MASK_ORT_DYLIB` | venv 内 libonnxruntime.so | 仅 `dynamic-ort` 构建需要 |
| `MASK_THREADS` | CPU 核数/2 | 推理线程数（2 核 → 0.56s/张，8 核 → 0.19s/张） |
| `MASK_NO_ARENA` | 1 | 关闭 CPU 内存池，RSS 更平稳 |

## 接口

`POST /process/`（multipart，字段 `file`）

query 参数（与 Python 版一致）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `output_format` | `jpg` | `jpg` / `png` |
| `style` | `mosaic` | `mosaic` 马赛克 / `blur` 高斯模糊 |
| `block` | 18 | 马赛克块大小 |
| `expand` | 8 | 检测框外扩像素 |
| `conf` | 0.1 | 检测阈值 |
| `mask_classes` | 空 | 要打码的类别，逗号分隔（`breast,anus,female_genital,male_genital`），空=全部 |
| `fallback` | `mosaic` | 兜底：`mosaic` / `blur` / `skip` / `off` |
| `fallback_cover_ratio` | 0 | 覆盖率兜底阈值，0=关闭 |
| `min_boxes` | 1 | 检出框数低于此值触发兜底 |

响应头：`X-Category`（分类结果）、`X-Cover`（检出覆盖率）、`X-Fallback`（是否兜底）、`X-Skip`（1=不发图）。

另有 `GET /health`、`GET /docs`、`GET /openapi.json`（插件用 `/docs` 探测服务可用性）。

## 验证

与 Python 版对拍（同图同参数）：

```
cmp_1448594: 平均差 0.84  最大差 32   差异>16 像素占比 0.12%
cmp_123456:  平均差 1.16  最大差 193  差异>16 像素占比 0.50%
```

差异来自 JPEG 编码器实现不同（Rust 用 `jpeg-encoder`，Python 用 OpenCV），打码区域与响应头完全一致。

## 实现要点

- 推理：`ort`（onnxruntime 绑定，`load-dynamic` 复用系统库）
- letterbox / 双线性 / 最近邻缩放：对齐 OpenCV 的半像素约定，裸字节切片实现
- YOLOv8 检测头解码 + IoU 0.45 NMS
- 分类与分割**并行**执行（`std::thread::scope`），总耗时取两者较大值
- 马赛克：矩形区域先缩到 `w/block × h/block` 再最近邻放大；模糊：σ 由 ksize 按 OpenCV 规则推导的分离式高斯

## 许可证

本项目代码为 **MIT**。

**模型权重**（`segmentation_model.onnx` / `classification_model.onnx`）版权归
[AutoHajimiMosaic](https://github.com/frinkleko/AutoHajimiMosaic) 作者，遵循 **GPL-3.0**，
**不在本仓库分发**，运行时通过环境变量指定路径加载。
