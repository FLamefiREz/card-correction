# Repository Guidelines

## Project Structure

- `main.cpp` runs the single-document demo. It reads, previews, and saves images. `CardCorrector` in `correct.cpp` owns the loaded ncnn model.
- `correct.h` declares the helpers implemented in `correct.cpp`, including detection decoding, coordinate transforms, and image cropping.
- Prepare the demo image at `example/1.jpg` and the ncnn model files `correction.param` and `correction.bin` locally. See the README for model locations. Git ignores all model and image files.

## Build and Run

Use CMake 3.27 or newer and a C++17 compiler. Install OpenCV 4 and ncnn for the same architecture as your build. Set `OpenCV_DIR` and `ncnn_DIR` to their CMake package directories when CMake cannot find them. Pass any vcpkg toolchain file during initial configuration.

From the repository root on Windows with a Visual Studio generator:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\cvDetectionCorrection.exe
```

Configure, build, and run the demo from the repository root. The demo saves `result.jpg` and displays a preview. Use `--no-gui example/1.jpg build/result.jpg` to run without preview or overwriting the root image.

## Coding Style

Use four-space indentation, PascalCase for types, and snake_case for functions and variables. Keep declarations and definitions in sync. Use short English comments to document coordinate spaces, tensor layouts, and other assumptions that affect results.

## Testing

Run the C++ regression tests with `ctest --test-dir build -C Release --output-on-failure`. The sample inference test is enabled when the local example image and both root model files exist at configure time. For geometry or orientation changes, also smoke-test a representative image.

## Commits and Pull Requests

Use short, imperative commit subjects, such as “Fix crop rotation for sideways cards.” In the PR, summarize the behavior change, link any related issue, and describe the build and smoke test you ran. Include before-and-after images for changes to visual output.

<!-- postman-cli:installation:start -->
Postman CLI is installed on this device, and can be used for API Engineering work.
<!-- postman-cli:installation:end -->
