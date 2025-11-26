# Aryorithm

### The Infrastructure for High-Performance AI

Aryorithm is on a mission to eliminate the friction between AI model training and production-grade, low-latency deployment. We build the core infrastructure that empowers developers and enterprises to serve optimized models at scale, with unparalleled speed and simplicity.

---
[**Get Started on Ignition-Hub**](https://hub.aryorithm.com/register) &nbsp;&nbsp;•&nbsp;&nbsp; [**Read the Documentation**](https://docs.aryorithm.com) &nbsp;&nbsp;•&nbsp;&nbsp; [**Join our Community Discord**](https://discord.gg/your-invite-code)
---

## The Aryorithm Pipeline: From Research to Real-Time

Our suite of tools forms a cohesive, end-to-end pipeline, automating the most complex parts of MLOps. We turn the painful "last mile" of deployment into a single, streamlined workflow.

1.  **Develop & Train:** Start with your trained PyTorch model (`.pth`).
2.  **Optimize & Convert:** Use our **XTorch** tool to convert your model into a highly-optimized TensorRT engine (`.engine`).
3.  **Deploy & Host:** Upload your engine to our **Ignition-Hub** cloud platform to instantly generate a secure, scalable REST API endpoint.
4.  **Integrate & Infer:** Call your new API from any application using our **XInfer** client SDKs (Python & C++) or any standard HTTP client.

---

## Our Core Repositories

Explore our open-source tools. We believe in building in the open and welcome community contributions.

| Repository | Description |
| :--- | :--- |
| **[aryorithm/xtorch](https://github.com/aryorithm/xtorch)** | The core PyTorch-to-TensorRT converter. Handles FP16/INT8 quantization, dynamic shapes, and custom plugins. |
| **[aryorithm/xinfer](https://github.com/aryorithm/xinfer)** | High-performance Python & C++ client SDKs for Ignition-Hub. Features async support and detailed timing info. |
| **[aryorithm/docs](https://github.com/aryorithm/docs)** | The source for our documentation website, built with Next.js. A great example of our philosophy in action. |
| **[aryorithm/examples](https://github.com/aryorithm/examples)** | A curated collection of end-to-end examples, from converting a YOLOv8 model to deploying a Transformer with C++. |

---

## Technology Spotlight

We solve hard engineering problems so you don't have to. Here's a look under the hood.

### XTorch: Intelligent Model Conversion

XTorch isn't just a script that calls `trtexec`. It's an intelligent conversion engine.
- **Graph Analysis:** Inspects your ONNX graph to automatically detect dynamic axes and suggest optimization profiles.
- **Smart Quantization:** The INT8 calibrator uses entropy minimization by default and can intelligently fall back to FP16 for layers that are sensitive to precision loss.
- **Plugin Management:** Seamlessly loads your custom TensorRT plugin libraries (`.so` files) during the build process.

```bash
# Example: INT8 conversion with a calibration dataset
xtorch convert --model yolov8.pth \
               --precision int8 \
               --calibration-data ./coco_calibration/ \
               --input-shape 1 3 640 640 \
               --output yolov8_int8.engine
```

### XInfer: The C++ Performance Edge

While our Python SDK is great for convenience, the C++ client is designed for bare-metal performance in demanding environments.
- **Zero-Overhead:** A modern, header-only C++17 library that avoids unnecessary abstractions.
- **Direct Memory Access:** Integrates with native libraries like OpenCV (`cv::Mat`) to prevent costly memory copies.
- **True Multithreading:** Designed for true parallelism in multi-core systems, unlike Python's GIL.

```cpp
// Example: C++ inference with XInfer
#include <xinfer/client.h>
#include <opencv2/opencv.hpp>

int main() {
    xinfer::Client client("YOUR_API_KEY");
    cv::Mat image = cv::imread("image.jpg");
    
    // Pre-process image...
    auto input_tensor = ...;

    auto output_tensor = client.infer("my-model:v1", input_tensor);

    // Post-process results...
    return 0;
}
```

---

## Get Started in 3 Steps

Experience the power of the Aryorithm pipeline today. Your first deployments are on us.

**1. Optimize with XTorch**

Install our open-source converter and optimize your first PyTorch model.```bash
pip install xtorch
xtorch convert --model your_model.pth --output model.engine --precision fp16
```

**2. Deploy on Ignition-Hub**

[**Sign up for a free account**](https://hub.aryorithm.com/register), create a new model, and upload your `model.engine` file.

**3. Infer with an API Call**

Instantly test your live endpoint with a simple `curl` command or our XInfer SDK.
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": ...}' \
  https://api.aryorithm.com/v1/infer/your-model
```

---

## Community & Support

-   **[Join our Discord](https://discord.gg/your-invite-code):** The best place to get help, share your projects, and chat directly with the engineering team.
-   **[Read the Blog](https://aryorithm.com/blog):** Deep dives, tutorials, and announcements.
-   **[Follow us on X/Twitter](https://twitter.com/aryorithm):** For real-time updates.
-   **Contribute:** Found a bug or have an idea? We welcome issues and pull requests on all our repositories.

***

**Aryorithm, Inc.** • Accelerating the Future of Artificial Intelligence.
