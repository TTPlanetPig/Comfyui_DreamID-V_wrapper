# ComfyUI DreamID-V Wrapper
[中文版](#中文说明) | [English](#english)

---

<a name="english"></a>
## English

A simple and efficient ComfyUI integration for **DreamID-V**.
**DreamID-V** is a model for **Identity-Preserving Video Generation**—it animates a reference image using motion from a pose video.

### Features
- **Simple Integration**: Designed for easy use within ComfyUI.
- **Selectable Precision**: Choose between `bf16`, `fp16`, etc. for best performance on your hardware.
- **Optimized**: Built-in acceleration for faster generation.
- **Flexible**: Various options for face recognition precision and methods.

### Installation
1.  Clone into `ComfyUI/custom_nodes`.
2.  `pip install -r requirements.txt`.
3.  **Download Models**:
    - `dreamidv.pth` -> `models/diffusion_models`.
    - `Wan2.1_VAE.pth` -> `models/vae`.
    - `umt5-xxl-enc-bf16.pth` -> `models/text_encoders`.

### Usage
1.  **Load Model**: Select precision (e.g. `bf16`).
2.  **Extract Pose**: Connect your video.
3.  **Generate**: Connect inputs and run.

### Screenshots
![Workflow Example](workflow.png)

---

<a name="中文说明"></a>
## 中文说明

**DreamID-V** 的 ComfyUI 简易整合包。
**DreamID-V** 是一款 **ID保持视频生成** 模型，它能根据一张参考图和一段骨架动作，生成保持人物身份且动作一致的视频。
主打简单易用，无需复杂配置即可体验高质量的人物视频生成。

### 主要功能
- **简单易用**：专为 ComfyUI 设计的简洁节点，即插即用。
- **精度可选**：支持 `bf16`、`fp16` 等多种精度选择，适配不同显卡。
- **人脸识别**：提供多种人脸识别精度和方式的选择。
- **极速生成**：内置多种加速优化，生成速度快。

### 安装方法
1.  在 `custom_nodes` 目录下克隆本项目。
2.  运行 `pip install -r requirements.txt` 安装依赖。
3.  **模型准备**：
    - `dreamidv.pth` 放入 `models/diffusion_models`。
    - `Wan2.1_VAE.pth` 放入 `models/vae`。
    - `umt5-xxl-enc-bf16.pth` 放入 `models/text_encoders`。

### 使用方法
1.  **加载模型**：选择合适的精度（如 RTX 30/40系列推荐 `bf16`）。
2.  **提取姿态**：使用姿态提取节点处理源视频。
3.  **生成视频**：连接各节点并开始生成。

### 节点截图
![Workflow Screenshot](workflow.png)

## Credits
Wrapper for [DreamID-V](https://github.com/bytedance/DreamID-V).
Original code by **Bytedance** and **The Alibaba Wan Team**.
