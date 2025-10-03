# AS-Transfer-Engine
AI-Powered Artistic Style Transfer 🎨

This project is a Python implementation of Neural Style Transfer (NST), an algorithm that uses deep learning to combine the content of one image with the artistic style of another. It leverages a pre-trained VGG-19 convolutional neural network to separate and recombine these visual elements.

This project was developed to combine skills from a Computer Science major and a Business minor, using Python and concepts from digital imaging.

## Features

- **VGG-19 Based Architecture**: Uses pre-trained VGG-19 model for feature extraction
- **Customizable Parameters**: Adjust style weight, content weight, learning rate, and optimization steps
- **GPU Support**: Automatically uses CUDA if available for faster processing
- **Multiple Style Layers**: Combines features from multiple convolutional layers for rich style transfer
- **Easy to Use**: Simple command-line interface and example scripts

## Requirements

- Python 3.8+
- PyTorch & Torchvision
- NumPy
- Matplotlib
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hdd5ps/AS-Transfer-Engine.git
cd AS-Transfer-Engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run style transfer using the command line:

```bash
python style_transfer.py --content path/to/content.jpg --style path/to/style.jpg --output output.png
```

#### Arguments:
- `--content`: Path to the content image (required)
- `--style`: Path to the style image (required)
- `--output`: Path to save the output image (default: output.png)
- `--steps`: Number of optimization steps (default: 2000)
- `--style-weight`: Weight for style loss (default: 1e6)
- `--content-weight`: Weight for content loss (default: 1)
- `--lr`: Learning rate for optimization (default: 0.003)

### Example Script

The repository includes an example script for quick testing:

1. Create an `examples` directory and add your images:
   - `examples/content.jpg` - Your content image
   - `examples/style.jpg` - Your style image

2. Run the example:
```bash
python example.py
```

The output will be saved to `output/stylized_image.png`.

### Python API

You can also use the style transfer functionality in your own Python scripts:

```python
from style_transfer import neural_style_transfer, save_result

# Perform style transfer
target = neural_style_transfer(
    content_image_path='path/to/content.jpg',
    style_image_path='path/to/style.jpg',
    num_steps=2000,
    style_weight=1e6,
    content_weight=1,
    learning_rate=0.003
)

# Save the result
save_result(target, 'output.png')
```

## How It Works

Neural Style Transfer works by optimizing an image to match the content of one image and the style of another:

1. **Feature Extraction**: The VGG-19 network extracts features from different layers
2. **Content Representation**: Deep layers capture the content structure
3. **Style Representation**: Gram matrices of multiple layers capture style patterns
4. **Optimization**: The target image is iteratively updated to minimize both content and style loss

The algorithm uses:
- **Content Loss**: Mean squared error between content features and target features
- **Style Loss**: Mean squared error between Gram matrices of style and target features
- **Total Loss**: Weighted combination of content and style losses

## Parameters Tuning

- **style_weight**: Higher values (1e6-1e7) create more stylized images
- **content_weight**: Usually kept at 1, increase to preserve more content details
- **steps**: More steps (2000-5000) produce better results but take longer
- **learning_rate**: Default 0.003 works well; adjust if optimization is unstable

## Technical Details

- **Model**: VGG-19 pre-trained on ImageNet
- **Content Layer**: conv4_2
- **Style Layers**: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
- **Optimizer**: Adam with configurable learning rate
- **Image Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original paper: ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
- VGG-19 model from PyTorch's torchvision.models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Examples

To get the best results:
1. Use high-quality images (both content and style)
2. Experiment with different style weights
3. Allow sufficient optimization steps (2000+ recommended)
4. Try different style images to achieve various artistic effects