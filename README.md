# AI-Powered Artistic Style Transfer 🎨

This project is a Python implementation of Neural Style Transfer (NST), an algorithm that uses deep learning to combine the content of one image with the artistic style of another. It leverages a pre-trained VGG-19 convolutional neural network to separate and recombine these visual elements.

This project was developed to combine skills from a Computer Science major and a Business minor, using Python and concepts from digital imaging.

## Technologies Used

* **Python 3.8+**
* **PyTorch & Torchvision**
* **NumPy**
* **Matplotlib**
* **Pillow (PIL)**

---

## Results Showcase

The model successfully transferred the style from a famous painting onto a photograph of the UVA Wise campus.


**Content Image**


**Style Image**


**Generated Output**

---

## Performance Analysis 🚀

A key goal was to demonstrate the performance gains from using NVIDIA GPU acceleration for AI tasks. The optimization loop was timed running on a standard CPU versus a GPU.

| Processor | Image Size | Iterations | Execution Time |
| :-------- | :--------- | :--------- | :------------- |
| **CPU** | 512x512    | 300        | ~14 minutes    |
| **GPU** | 512x512    | 300        | **~28 seconds** |

The results show a **dramatic speedup of over 30x**. This is because the tensor computations required for the neural network can be performed in parallel on the GPU's thousands of cores, making it far more efficient than the sequential processing of a CPU.

---

## Business Application

Drawing on principles from my Business minor and innovation coursework, this project can be framed as a prototype for a **"Style-as-a-Service" (SaaS) application**.

* **Target Market:** Social media content creators, digital artists, and marketing agencies seeking unique, eye-catching visuals.
* **Value Proposition:** An easy-to-use cloud platform that allows users to apply any artistic style to their photos or videos without needing technical expertise or powerful hardware.
* **Monetization Strategy:** A tiered subscription model. A free tier could offer low-resolution images with a watermark, while premium tiers could provide high-resolution outputs, video processing, and API access for businesses. This model provides a low barrier to entry while creating a path to revenue from power users and commercial clients.
