from ultralytics import YOLO
from config import PARAMS
# Load a model
work_dir = PARAMS['work_dir']
model = YOLO(f"{work_dir}/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
