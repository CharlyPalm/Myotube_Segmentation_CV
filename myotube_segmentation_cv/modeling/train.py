# myotube_segmentation_cv/train.py
import torch
from ultralytics import YOLO
from pathlib import Path
# Código basado en la implementación de Ultralytics YOLOv11 # Fuente: https://github.com/ultralytics/yolov11

# Define paths and parameters directly in the script
MODEL_PATH = Path("../../models/yolo11n-seg.pt")  # Path to the model weights
MODEL_YAML = Path("yolo11n-seg.yaml")  # Path to the model YAML config
DATASET_PATH = Path("../models/dataset.yaml")  # Path to dataset YAML

# Hyperparameters (can be adjusted here)
EPOCHS = 500  # Number of epochs to train
IMGSZ = 1024  # Image size for training
BATCH_SIZE = 12  # Batch size
DEVICE = 0  # Set the device (GPU 0)
LR0 = 1e-4  # Initial learning rate
FREEZE = 16  # Number of layers to freeze
PRETRAINED = True  # Whether to use pretrained weights
OVERLAP_MASK = False  # Whether to use overlap mask
PATIENCE = 100  # Patience for early stopping

# Print memory summary before training to check GPU memory
#torch.cuda.empty_cache()
#print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Initialize and load the YOLO model
model = YOLO(MODEL_YAML).load(MODEL_PATH)

# Train the model
results = model.train(
    data=DATASET_PATH, 
    epochs=EPOCHS, 
    imgsz=IMGSZ, 
    batch=BATCH_SIZE,  
    device=DEVICE, 
    patience=PATIENCE, 
    lr0=LR0, 
    freeze=FREEZE, 
    pretrained=PRETRAINED, 
    overlap_mask=OVERLAP_MASK
)

# Optionally print the results of the training
print(results)

# Print the memory summary again after training
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
