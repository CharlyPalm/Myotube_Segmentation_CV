# myotube_segmentation_cv/predict.py
from ultralytics import YOLO
from pathlib import Path
# Código basado en la implementación de Ultralytics YOLOv11 # Fuente: https://github.com/ultralytics/yolov11

def predict_image(model_path: Path, 
                  image_path: Path, 
                  confidence_threshold: float = 0.10, 
                  iou_threshold: float = 0.01, 
                  device: int = 0, 
                  visualize: bool = False, 
                  save_txt: bool = False):
    """
    Function to make predictions on a single image using a YOLO model.
    
    Args:
        model_path (Path): Path to the trained YOLO model weights.
        image_path (Path): Path to the image for inference.
        confidence_threshold (float): Confidence threshold for predictions.
        iou_threshold (float): IOU threshold for predictions.
        device (int): Device number to use (e.g., 0 for GPU).
        visualize (bool): Whether to display the results with visualizations.
        save_txt (bool): Whether to save predictions as a text file.
    
    Returns:
        results: The results from the YOLO prediction.
    """
    # Initialize and load the YOLO model
    model = YOLO(model_path)

    # Perform inference (prediction) on the image
    results = model.predict(
        image_path, 
        save=True, 
        conf=confidence_threshold, 
        iou=iou_threshold, 
        device=device, 
        visualize=visualize,  # Change to True to see the visualization
        save_txt=save_txt  # Whether to save the predictions as a text file
    )
    
    return results
