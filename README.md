# card-correction
证件、单据等矫正

## 功能描述

示例检测图中置信度最高的单张证件或单据，并进行透视裁剪和方向矫正。没有检测达到阈值时保存原图。批量处理多张图片时，可以复用一个 `CardCorrector` 实例。

同一张图中输出多张证件需要对多个候选框分别筛选、去重和裁剪，并返回多张结果。调整候选数量只会影响检测阶段。

## 构建

需要 CMake 3.27 或更新版本、C++17 编译器、[OpenCV 4](https://opencv.org/releases/) 和 [ncnn](https://github.com/Tencent/ncnn/releases)。从仓库根目录运行。以下示例使用 Windows PowerShell 和 Visual Studio 生成器，请替换为本机依赖路径。

```powershell
cmake -S . -B build -A x64 "-DOpenCV_DIR=C:/path/to/opencv/build" "-Dncnn_DIR=C:/path/to/ncnn/x64/lib/cmake/ncnn"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

若用 vcpkg 安装依赖，在首次配置时传入 `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`。使用动态库时，将所用依赖的 DLL 目录加入 `PATH`。

## 运行

```powershell
.\build\Release\cvDetectionCorrection.exe
.\build\Release\cvDetectionCorrection.exe --no-gui example/1.jpg build/result.jpg
```

模型和图片需要在本地准备。把 ncnn 格式的 `correction.param` 和 `correction.bin` 放在仓库根目录或 `model/`。如果 `model/` 中两个文件都存在，程序优先使用该目录，也可以通过第三个位置参数指定目录。

默认读取 `example/1.jpg`，保存 `result.jpg` 并打开预览。也可以传入自己的图片路径。不准备模型和样图时，CTest 仍会运行单元测试。配置 CMake 时，如果仓库根目录的模型文件和 `example/1.jpg` 已就绪，CTest 还会运行样图推理测试。

## 参考网站

https://www.modelscope.cn/models/iic/cv_resnet18_card_correction/summary 感谢魔塔开源的模型

https://github.com/Tencent/ncnn 感谢nihui开源的框架支持

https://github.com/FeiGeChuanShu/ncnn_paddleocr 后续证件识别
