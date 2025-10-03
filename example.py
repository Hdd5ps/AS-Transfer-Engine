"""
Example script for running Neural Style Transfer
"""

from style_transfer import neural_style_transfer, save_result
import os


def run_example():
    """
    Run a simple example of neural style transfer.
    Note: You'll need to provide your own content and style images.
    """
    # Check if example images exist
    if not os.path.exists('examples'):
        print("Creating examples directory...")
        os.makedirs('examples', exist_ok=True)
        print("\nPlease add your images to the 'examples' directory:")
        print("  - examples/content.jpg (your content image)")
        print("  - examples/style.jpg (your style image)")
        return
    
    content_path = 'examples/content.jpg'
    style_path = 'examples/style.jpg'
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print("Example images not found!")
        print("\nPlease add your images to the 'examples' directory:")
        print(f"  - {content_path} (your content image)")
        print(f"  - {style_path} (your style image)")
        return
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    print("Running Neural Style Transfer...")
    print(f"Content: {content_path}")
    print(f"Style: {style_path}")
    print()
    
    # Run style transfer with default parameters
    target = neural_style_transfer(
        content_image_path=content_path,
        style_image_path=style_path,
        num_steps=2000,
        style_weight=1e6,
        content_weight=1,
        learning_rate=0.003,
        show_every=400
    )
    
    # Save the result
    output_path = 'output/stylized_image.png'
    save_result(target, output_path)
    
    print(f"\nSuccess! Check {output_path} for the result.")


if __name__ == "__main__":
    run_example()
