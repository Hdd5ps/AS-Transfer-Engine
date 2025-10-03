"""
Neural Style Transfer Implementation
Using VGG-19 pre-trained model to transfer artistic style from one image to another.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy


class VGGFeatureExtractor(nn.Module):
    """
    VGG-19 model for extracting features at specific layers.
    """
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # Load pre-trained VGG-19 model
        vgg = models.vgg19(pretrained=True).features
        
        # Define layers for content and style extraction
        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
            '28': 'conv5_1'
        }
        
        # Split VGG into sections at the layers we care about
        self.model = nn.Sequential()
        for name, layer in vgg._modules.items():
            self.model.add_module(name, layer)
            
        # Freeze all VGG parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        """
        Extract features from multiple layers.
        """
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features


def load_image(image_path, max_size=400, shape=None):
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size for the image
        shape: Optional shape to resize to
    
    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    # Resize image
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension
    image = transform(image).unsqueeze(0)
    
    return image


def im_convert(tensor):
    """
    Convert a tensor to an image for display.
    
    Args:
        tensor: Image tensor
    
    Returns:
        NumPy array representing the image
    """
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    
    return image


def gram_matrix(tensor):
    """
    Calculate the Gram matrix for style representation.
    
    Args:
        tensor: Feature tensor
    
    Returns:
        Gram matrix
    """
    batch_size, channels, height, width = tensor.size()
    
    # Reshape to (channels, height*width)
    tensor = tensor.view(channels, height * width)
    
    # Calculate gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram


def calculate_content_loss(target_features, content_features):
    """
    Calculate content loss.
    
    Args:
        target_features: Features from the target image
        content_features: Features from the content image
    
    Returns:
        Content loss value
    """
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    return content_loss


def calculate_style_loss(target_features, style_features, style_weights):
    """
    Calculate style loss.
    
    Args:
        target_features: Features from the target image
        style_features: Features from the style image
        style_weights: Weights for each style layer
    
    Returns:
        Style loss value
    """
    style_loss = 0
    
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        
        style_feature = style_features[layer]
        style_gram = gram_matrix(style_feature)
        
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        
        batch_size, channels, height, width = target_feature.shape
        style_loss += layer_style_loss / (channels * height * width)
    
    return style_loss


def neural_style_transfer(content_image_path, style_image_path, 
                         num_steps=2000, style_weight=1e6, content_weight=1,
                         learning_rate=0.003, show_every=400):
    """
    Perform neural style transfer.
    
    Args:
        content_image_path: Path to content image
        style_image_path: Path to style image
        num_steps: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        learning_rate: Learning rate for optimizer
        show_every: Show progress every N steps
    
    Returns:
        Generated image tensor
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load images
    content = load_image(content_image_path).to(device)
    style = load_image(style_image_path, shape=content.shape[-2:]).to(device)
    
    # Display original images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(im_convert(content))
    ax1.set_title('Content Image')
    ax1.axis('off')
    
    ax2.imshow(im_convert(style))
    ax2.set_title('Style Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('input_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Initialize target image with content image
    target = content.clone().requires_grad_(True).to(device)
    
    # Load VGG model
    vgg = VGGFeatureExtractor().to(device)
    
    # Define style layer weights
    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.1
    }
    
    # Extract features from content and style images
    content_features = vgg(content)
    style_features = vgg(style)
    
    # Optimizer
    optimizer = optim.Adam([target], lr=learning_rate)
    
    # Optimization loop
    print("Starting style transfer...")
    for step in range(1, num_steps + 1):
        # Extract features from target image
        target_features = vgg(target)
        
        # Calculate losses
        content_loss = calculate_content_loss(target_features, content_features)
        style_loss = calculate_style_loss(target_features, style_features, style_weights)
        
        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if step % show_every == 0 or step == 1:
            print(f"Step {step}/{num_steps}")
            print(f"  Total Loss: {total_loss.item():.4f}")
            print(f"  Content Loss: {content_loss.item():.4f}")
            print(f"  Style Loss: {style_loss.item():.4f}")
            print()
    
    print("Style transfer complete!")
    return target


def save_result(target, output_path='output.png'):
    """
    Save the generated image.
    
    Args:
        target: Generated image tensor
        output_path: Path to save the image
    """
    result = im_convert(target)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.title('Generated Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Result saved to {output_path}")


def main():
    """
    Main function to run style transfer with example usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True,
                       help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                       help='Path to style image')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Path to save output image')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Number of optimization steps')
    parser.add_argument('--style-weight', type=float, default=1e6,
                       help='Weight for style loss')
    parser.add_argument('--content-weight', type=float, default=1,
                       help='Weight for content loss')
    parser.add_argument('--lr', type=float, default=0.003,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Perform style transfer
    target = neural_style_transfer(
        args.content,
        args.style,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        learning_rate=args.lr
    )
    
    # Save result
    save_result(target, args.output)


if __name__ == "__main__":
    main()
