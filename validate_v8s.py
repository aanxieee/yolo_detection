import os
import multiprocessing as mp
from ultralytics import YOLO

# Set the environment variable to allow duplicate OpenMP runtime initialization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    os.chdir(r'D:\aanya')

    # Load the trained model
    model = YOLO(r"D:\aanya\yolov8s.pt")  # Using raw string to handle backslashes correctly

    # Customize validation settings
    validation_results = model.val(data=r"D:\aanya\data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
    print(f"Validation results: {validation_results}")

if __name__ == "__main__":
    # Required on Windows to prevent multiprocessing errors
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    main()

