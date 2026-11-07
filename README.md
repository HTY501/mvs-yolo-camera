# 工业相机图像采集与 YOLO 实时识别

## 1. 实验环境
- **相机型号**：MV-CS016-10UC（序列号：DA4302122）
- **操作系统**：Ubuntu 22.04 (VMware 虚拟机)
- **依赖库**：MVS SDK (V4.5.0), OpenCV 4.5.4, OpenVINO 2024.6.0
- **推理设备**：CPU
- **模型格式**：OpenVINO (.xml, .bin)

## 2. 程序运行方法
```bash
mkdir build && cd build
cmake ..
make
# 运行前需设置 MVS 动态库路径
export LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH
./mvs_yolo
```
3. 核心处理流程
使用 MVS SDK 枚举并连接相机，关闭自动曝光和自动增益。

启动取流，通过 MV_CC_GetImageBuffer 获取一帧图像。

将原生 BayerRG8 数据转换为 OpenCV 的 BGR8 格式（由于 OpenCV 与海康 SDK 的 Bayer 通道定义差异，代码中进行了 R 和 B 通道互换处理）。

对图像进行 Resize 和归一化，送入 OpenVINO 进行推理。

解析 YOLOv8 输出，进行 NMS（非极大值抑制）过滤。

在图像上绘制检测框、置信度以及 4 个关键点，并实时显示。

调用 MV_CC_FreeImageBuffer 归还内存，避免泄漏。

4. 实验记录与数据分析
4.1 曝光时间测试 (固定增益 0dB)
500 μs：画面较暗，但运动物体拖影很小。

1000 μs：画面亮度适中，满足常规识别需求。

3000 μs：画面明显变亮，但快速移动装甲板时，拖影严重，导致 YOLO 识别框抖动甚至漏检。

4.2 增益测试 (固定曝光 1000μs)
0 dB：画面正常，暗部较为干净。

6 dB：画面亮度提升。

12 dB：画面整体变亮，但放大观察暗部背景，出现了明显的彩色噪点，影响 YOLO 的特征提取。

4.3 图像格式与性能
相机原生输出为 BayerRG8（单通道），需转换为 BGR8（三通道）。

转换后帧率大约为 30 FPS（请根据你的实际观察修改），YOLO 推理单帧耗时约 15 ms（请根据你的实际修改）。

5. 遇到问题与异常处理
问题 1：OpenCV DNN 无法加载 ONNX 模型。报错 Split 算子不支持。解决：放弃 OpenCV DNN，改用 Intel OpenVINO 框架进行推理，完美兼容且提速。

问题 2：图像红蓝颜色反转。解决：在 cv::cvtColor 后，通过 split 和 merge 强制交换 R 和 B 通道。

异常处理：在 while 循环中加入了超时判定（ret == MV_OK）。如果中途拔掉 USB 线，程序会打印错误码并安全退出，不会发生内存崩溃。
