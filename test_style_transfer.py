"""
Tests for Neural Style Transfer implementation
"""

import unittest
import torch
import numpy as np
from PIL import Image
import os
import tempfile

from style_transfer import (
    VGGFeatureExtractor,
    load_image,
    im_convert,
    gram_matrix,
    calculate_content_loss,
    calculate_style_loss
)


class TestStyleTransfer(unittest.TestCase):
    """Test cases for style transfer functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device("cpu")  # Use CPU for testing
        
    def test_vgg_feature_extractor(self):
        """Test VGG feature extractor initialization"""
        model = VGGFeatureExtractor()
        self.assertIsInstance(model, torch.nn.Module)
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        features = model(dummy_input)
        
        # Check that expected layers are present
        expected_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        for layer in expected_layers:
            self.assertIn(layer, features)
    
    def test_load_image(self):
        """Test image loading and preprocessing"""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image = Image.new('RGB', (100, 100), color='red')
            test_image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Load the image
            loaded = load_image(tmp_path, max_size=50)
            
            # Check tensor properties
            self.assertEqual(len(loaded.shape), 4)  # Should be (batch, channels, height, width)
            self.assertEqual(loaded.shape[0], 1)    # Batch size should be 1
            self.assertEqual(loaded.shape[1], 3)    # RGB channels
            
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_im_convert(self):
        """Test tensor to image conversion"""
        # Create a random tensor
        tensor = torch.randn(1, 3, 100, 100)
        
        # Convert to image
        image = im_convert(tensor)
        
        # Check output properties
        self.assertEqual(len(image.shape), 3)  # (height, width, channels)
        self.assertEqual(image.shape[2], 3)    # RGB channels
        self.assertTrue(np.all(image >= 0))    # Values should be >= 0
        self.assertTrue(np.all(image <= 1))    # Values should be <= 1
    
    def test_gram_matrix(self):
        """Test Gram matrix calculation"""
        # Create a test tensor
        tensor = torch.randn(1, 64, 10, 10)
        
        # Calculate Gram matrix
        gram = gram_matrix(tensor)
        
        # Check output shape
        self.assertEqual(gram.shape, (64, 64))
        
        # Gram matrix should be symmetric
        self.assertTrue(torch.allclose(gram, gram.t(), atol=1e-5))
    
    def test_content_loss(self):
        """Test content loss calculation"""
        # Create dummy features
        target_features = {
            'conv4_2': torch.randn(1, 512, 28, 28)
        }
        content_features = {
            'conv4_2': torch.randn(1, 512, 28, 28)
        }
        
        # Calculate loss
        loss = calculate_content_loss(target_features, content_features)
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Loss should be non-negative
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_style_loss(self):
        """Test style loss calculation"""
        # Create dummy features
        target_features = {
            'conv1_1': torch.randn(1, 64, 224, 224),
            'conv2_1': torch.randn(1, 128, 112, 112)
        }
        style_features = {
            'conv1_1': torch.randn(1, 64, 224, 224),
            'conv2_1': torch.randn(1, 128, 112, 112)
        }
        style_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.8
        }
        
        # Calculate loss
        loss = calculate_style_loss(target_features, style_features, style_weights)
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Loss should be non-negative
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_identical_images_have_zero_content_loss(self):
        """Test that identical images have zero content loss"""
        features = {
            'conv4_2': torch.randn(1, 512, 28, 28)
        }
        
        # Same features should give zero loss
        loss = calculate_content_loss(features, features)
        self.assertAlmostEqual(loss.item(), 0, places=5)


if __name__ == '__main__':
    unittest.main()
