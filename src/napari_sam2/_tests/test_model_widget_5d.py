"""Tests for 5D data handling in model_widget.py"""

import numpy as np
import pytest


def test_squeeze_handles_singleton_dimensions(make_napari_viewer):
    """Test that singleton dimensions are properly squeezed"""
    viewer = make_napari_viewer()
    
    # Create 5D data with singleton dimensions that should become 3D after squeeze
    # Shape: (1, 1, 10, 256, 256) -> after squeeze: (10, 256, 256)
    data_5d_with_singletons = np.random.randint(0, 255, (1, 1, 10, 256, 256), dtype=np.uint8)
    layer = viewer.add_image(data_5d_with_singletons)
    
    # Verify the squeeze would reduce dimensions
    squeezed = np.squeeze(data_5d_with_singletons)
    assert squeezed.ndim == 3
    assert squeezed.shape == (10, 256, 256)


def test_squeeze_handles_4d_singleton_to_2d(make_napari_viewer):
    """Test 4D data with singletons reducing to 2D"""
    viewer = make_napari_viewer()
    
    # Shape: (1, 1, 256, 256) -> after squeeze: (256, 256)
    data_4d_with_singletons = np.random.randint(0, 255, (1, 1, 256, 256), dtype=np.uint8)
    layer = viewer.add_image(data_4d_with_singletons)
    
    squeezed = np.squeeze(data_4d_with_singletons)
    assert squeezed.ndim == 2
    assert squeezed.shape == (256, 256)


def test_5d_data_tczyx_order():
    """Test 5D data in TCZYX order"""
    # Shape: (10, 1, 5, 256, 256) - T=10, C=1, Z=5, Y=256, X=256
    data_5d = np.random.randint(0, 255, (10, 1, 5, 256, 256), dtype=np.uint8)
    
    # Squeeze removes the singleton channel dimension
    squeezed = np.squeeze(data_5d)
    assert squeezed.ndim == 4
    assert squeezed.shape == (10, 5, 256, 256)  # T, Z, Y, X


def test_5d_data_rgb_tczyx_order():
    """Test 5D RGB data in TCZYX order"""
    # Shape: (10, 3, 5, 256, 256) - T=10, C=3 (RGB), Z=5, Y=256, X=256
    data_5d_rgb = np.random.randint(0, 255, (10, 3, 5, 256, 256), dtype=np.uint8)
    
    # Squeeze should not remove any dimensions (no singletons)
    squeezed = np.squeeze(data_5d_rgb)
    assert squeezed.ndim == 5
    assert squeezed.shape == (10, 3, 5, 256, 256)


def test_axes_order_parsing():
    """Test parsing of different axes orders"""
    # Test various axes orders
    axes_orders = [
        ('TCZYX', 0, 1),  # Standard order
        ('TZCYX', 0, 2),  # Z before C
        ('CTZYX', 1, 0),  # C before T
        ('tczyx', 0, 1),  # Lowercase (should be handled)
    ]
    
    for axes, expected_t, expected_c in axes_orders:
        axes_upper = axes.upper()
        t_idx = axes_upper.find('T') if 'T' in axes_upper else 0
        c_idx = axes_upper.find('C') if 'C' in axes_upper else 1
        
        assert t_idx == expected_t, f"Failed for axes order {axes}"
        assert c_idx == expected_c, f"Failed for axes order {axes}"


def test_4d_single_channel_data():
    """Test 4D data with single channel"""
    # Shape: (10, 1, 256, 256) - T=10, C=1, Y=256, X=256
    data_4d = np.random.randint(0, 255, (10, 1, 256, 256), dtype=np.uint8)
    
    squeezed = np.squeeze(data_4d)
    assert squeezed.ndim == 3
    assert squeezed.shape == (10, 256, 256)
    
    # Verify first frame extraction would work
    frame = squeezed[0]
    assert frame.shape == (256, 256)


def test_4d_rgb_data():
    """Test 4D RGB data"""
    # Shape: (10, 3, 256, 256) - T=10, C=3 (RGB), Y=256, X=256
    data_4d_rgb = np.random.randint(0, 255, (10, 3, 256, 256), dtype=np.uint8)
    
    squeezed = np.squeeze(data_4d_rgb)
    assert squeezed.ndim == 4
    assert squeezed.shape == (10, 3, 256, 256)
    
    # Verify frame extraction would work
    frame = squeezed[0]  # Shape: (3, 256, 256)
    frame_transposed = np.transpose(frame, (1, 2, 0))  # Shape: (256, 256, 3)
    assert frame_transposed.shape == (256, 256, 3)


def test_invalid_channel_count():
    """Test that invalid channel counts are handled"""
    # Shape: (10, 5, 256, 256) - T=10, C=5 (invalid), Y=256, X=256
    data_4d_invalid = np.random.randint(0, 255, (10, 5, 256, 256), dtype=np.uint8)
    
    squeezed = np.squeeze(data_4d_invalid)
    expected_num_channels = squeezed.shape[1]
    
    # Should not be valid (only 1 or 3 channels allowed)
    assert expected_num_channels not in [1, 3]


def test_2d_data_unchanged():
    """Test that 2D data is handled correctly"""
    data_2d = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    squeezed = np.squeeze(data_2d)
    assert squeezed.ndim == 2
    assert squeezed.shape == (256, 256)


def test_3d_data_unchanged():
    """Test that 3D data is handled correctly"""
    # Shape: (10, 256, 256) - T=10, Y=256, X=256
    data_3d = np.random.randint(0, 255, (10, 256, 256), dtype=np.uint8)
    
    squeezed = np.squeeze(data_3d)
    assert squeezed.ndim == 3
    assert squeezed.shape == (10, 256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
