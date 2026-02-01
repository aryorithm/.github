# Aryorithm

### The Infrastructure for High-Performance AI

Aryorithm is on a mission to eliminate the friction between AI model training and production-grade, low-latency deployment. We build the core infrastructure that empowers developers and enterprises to serve optimized models at scale, with unparalleled speed and simplicity.

---
- [**Get Started on Ignition-Hub**](https://aryorithm.com/register) 
- [**Read XInfer Documentation**](https://aryorithm.com/xinfer/docs)  
- [**Read XTorch Documentation**](https://aryorithm.com/xtorch/docs)
- [**Join our Community Discord**](https://discord.gg/your-invite-code)
---

## The Aryorithm Pipeline: From Research to Real-Time

Our suite of tools forms a cohesive, end-to-end pipeline, automating the most complex parts of MLOps. We turn the painful "last mile" of deployment into a single, streamlined workflow.

1.  **Develop & Train:** Start with your trained PyTorch model (`.pth`).
2.  **Optimize & Convert:** Use our open-source **XTorch** tool to convert your model into a highly-optimized TensorRT engine (`.engine`). This step alone can provide a 5-10x performance boost.
3.  **Deploy & Host:** Upload your engine to our **Ignition-Hub** cloud platform to instantly generate a secure, scalable, and serverless REST API endpoint. No Docker, no Kubernetes, no servers to manage.
4.  **Integrate & Infer:** Call your new API from any application using our high-performance **XInfer** client SDKs (Python & C++) or any standard HTTP client.

---

## Why Aryorithm? The End of Deployment Headaches

If you've ever tried to deploy a machine learning model, you've likely faced these challenges. Here's how we solve them.

| Pain Point | The Old Way | The Aryorithm Way |
| :--- | :--- | :--- |
| **Performance** | Python server (Flask/FastAPI) hits GIL bottlenecks; manual TensorRT conversion is complex and error-prone. | **XTorch** automates TensorRT optimization. **Ignition-Hub** serves models from a high-performance C++ core. |
| **Cost** | A dedicated GPU instance runs 24/7, costing hundreds of dollars per month, even when idle. | **Ignition-Hub** is serverless. You pay only for the milliseconds of GPU time you actually use. No traffic, no cost. |
| **Scalability** | You are responsible for setting up auto-scaling groups, load balancers, and managing infrastructure. | **Ignition-Hub** scales from zero to millions of requests automatically. We handle the infrastructure for you. |
| **Complexity** | Wrestling with NVIDIA drivers, CUDA versions, Dockerfiles, and Kubernetes configurations. | A `pip install xtorch`, a single command, and a file upload. Your focus stays on your model, not on DevOps. |

---

## Our Core Repositories

Explore our open-source tools. We believe in building in the open and welcome community contributions.

| Repository | Description |
| :--- | :--- |
| **[aryorithm/xtorch](https://github.com/aryorithm/xtorch)** | The core PyTorch-to-TensorRT converter. Handles FP16/INT8 quantization, dynamic shapes, and custom plugins. |
| **[aryorithm/xinfer](https://github.com/aryorithm/xinfer)** | High-performance Python & C++ client SDKs for Ignition-Hub. Features async support, gRPC streaming, and detailed timing info. |
| **[aryorithm/docs](https://github.com/aryorithm/docs)** | The source for our documentation website, built with Next.js. A great example of our philosophy in action. |
| **[aryorithm/examples](https://github.com/aryorithm/examples)** | A curated collection of end-to-end examples, from converting a YOLOv8 model to deploying a Transformer with C++. |
| **[aryorithm/engines](https://github.com/aryorithm/engines)** | List of compiled engines. |

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

### Ignition-Hub: A True Serverless GPU Platform
  
Ignition-Hub is more than just a model host. It's a purpose-built inference machine.
- **Event-Driven Scaling:** We don't just scale up VMs. Our architecture routes requests to a pool of warm, ready-to-go GPU workers. New workers are provisioned in milliseconds, not minutes, to handle traffic bursts.
- **Smart Queuing:** Pro and Enterprise requests get priority routing to minimize queue wait times, ensuring consistently low latency.
- **Secure by Design:** Models are encrypted at rest (AES-256) and in transit (TLS 1.3). Inference jobs run in fully isolated, sandboxed environments.

---

## Popular Use Cases

Aryorithm is built for any application that needs fast, reliable AI inference.
- **Real-Time Video Analytics:** Process streams from security cameras or drones with consistently low latency.
- **Interactive Web Applications:** Power features like AI-driven search, content generation, or image editing directly in your web app.
- **Robotics & Automotive:** Deploy perception models on embedded systems using engines optimized by XTorch and accessed via the XInfer C++ SDK.
- **Scientific Computing:** Accelerate complex simulations by offloading computational kernels to the Ignition-Hub platform.

---

## Get Started in 3 Steps

Experience the power of the Aryorithm pipeline today. Your first deployments are on us.

**1. Optimize with XTorch**
Install our open-source converter and optimize your first PyTorch model.
```bash
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

## Our Philosophy

1.  **Developers First:** Our tools are built for developers, by developers. We prioritize clean APIs, great documentation, and a seamless workflow.
2.  **Performance is a Feature:** We believe that speed and efficiency are not afterthoughts—they are core requirements for successful AI products.
3.  **Open Core:** We are committed to keeping our client-side tools, XTorch and XInfer, free and open-source forever. Our business is our cloud platform, not the tools you run on your machine.

---

## Our Team

Meet the core team behind Aryorithm. We are passionate about building the next generation of tools for AI developers.

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kamisaberi">
        <img src="https://github.com/kamisaberi.png?size=100" width="50px;" alt="Kami"/>
        <br />
        <sub><b>Kamran Saberifard</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/SbhnNP">
        <img src="https://github.com/SbhnNP.png?size=100" width="50px;" alt="Full Name 3"/>
        <br />
        <sub><b>Sobhan Nikpour</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/parrnn">
        <img src="https://github.com/parrnn.png?size=100" width="50px;" alt="Full Name 3"/>
        <br />
        <sub><b>Pariya Ranji</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/adelbozorgbashar">
        <img src="https://github.com/adelbozorgbashar.png?size=100" width="50px;" alt="Full Name 3"/>
        <br />
        <sub><b>Adel Bozorg Bashar</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/vidaentezar">
        <img src="https://github.com/vidaentezar.png?size=100" width="50px;" alt="Full Name 3"/>
        <br />
        <sub><b>Adel Bozorg Bashar</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ParsaDokhtMohammadi">
        <img src="https://github.com/ParsaDokhtMohammadi.png?size=100" width="50px;" alt="Full Name 3"/>
        <br />
        <sub><b>Parsa Dokht Mohammadi</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Abol-khls">
        <img src="https://github.com/Abol-khls.png?size=100" width="50px;" alt="Full Name 3"/>
        <br />
        <sub><b>Abolfazl Kholousi</b></sub>
      </a>
    </td>
  </tr>
</table>



---
## Community & Support

We're building the future of AI deployment, and we want you to be a part of it.
-   **[Join our Discord](https://discord.gg/your-invite-code):** The best place to get help, share your projects, and chat directly with the engineering team.
-   **[Read the Latest News](https://aryorithm.com/news):** Deep dives, tutorials, and announcements.
-   **[Follow us on X/Twitter](https://twitter.com/aryorithm):** For real-time updates.
-   **Contribute:** Found a bug or have an idea? We welcome issues and pull requests on all our repositories. Check out our `CONTRIBUTING.md` guides to get started.
---

***

**Aryorithm, Inc.** • Accelerating the Future of Artificial Intelligence.
