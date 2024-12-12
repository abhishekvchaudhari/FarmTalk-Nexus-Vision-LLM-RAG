from ultralytics import YOLO

# Load Yolov8n
model = YOLO('yolov8n.pt') 

# Train the model
model.train(
    data='plant_disease.yaml',
    epochs=10,
    imgsz=256,    
    batch=64,       
    device=0,
    save=True,
)

