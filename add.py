import os
from paddleocr import PaddleOCR

# Create models directory
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Download models
ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='en', 
    download_dir=models_dir
)