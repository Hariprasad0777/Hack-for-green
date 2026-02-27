import sys
import os
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path


from offroad_ai.pipeline.pathway_engine import perform_inference, init_model
from offroad_ai.core import config

@pytest.fixture
def mock_image_data():
    return {
        'path': 'test_img.jpg',
        'data': np.zeros((100, 100, 3), dtype=np.uint8).tobytes()
    }

@patch('pipeline.pathway_engine.MODEL')
@patch('pipeline.pathway_engine.TRANSFORM')
@patch('cv2.imdecode')
@patch('cv2.imwrite')
def test_perform_inference_flow(mock_imwrite, mock_imdecode, mock_transform, mock_model, mock_image_data):
    """Verify the inference flow inside the Pathway worker."""
    # Setup mocks
    mock_imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_transform.return_value = {'image': torch.randn(3, config.IMG_SIZE, config.IMG_SIZE)}
    
    # Mock model output (1 batch, 6 classes, IMG_SIZE, IMG_SIZE)
    mock_model.return_value = torch.randn(1, config.NUM_CLASSES, config.IMG_SIZE, config.IMG_SIZE)
    
    # Execute
    result = perform_inference(mock_image_data)
    
    # Assertions
    assert "filename" in result
    assert "ground_percentage" in result
    assert "obstacle_percentage" in result
    assert result["filename"] == "test_img.jpg"
    assert mock_model.called
    assert mock_imwrite.called

def test_model_lazy_init():
    """Verify that init_model initializes the global MODEL variable."""
    with patch('pipeline.pathway_engine.get_model') as mock_get_model:
        mock_get_model.return_value = MagicMock()
        init_model()
        from offroad_ai.pipeline import pathway_engine
        assert pathway_engine.MODEL is not None
