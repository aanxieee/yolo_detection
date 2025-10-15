import os
import multiprocessing as mp
from ultralytics import YOLO

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    # Change to the directory containing the model and data
    os.chdir(r'D:\Amarjeet_CrackModel\Crack_Final')

    # Load the YOLO model
    model = YOLO(r"D:\Amarjeet_CrackModel\Crack_Final\yolov9m.pt")

    # Train the model with specified settings
    model.train(
        data=r"D:\Amarjeet_CrackModel\Crack_Final\data.yaml", 
        epochs=150, 
        batch=16, 
        imgsz=640, 
        device="0",
        project=r"D:\Amarjeet_CrackModel\Crack_Final\detect\run",
        # resume=r"D:\Amarjeet_CrackModel\Crack_Final\detect\run\train2\weights\last.pt",  # Path to the checkpoint file
        # name="train"
    )
    print("Training complete.")

    # Uncomment to perform validation after training if desired
    # validation_results = model.val(data=r"D:\aanya\data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
    # print(f"Validation results: {validation_results}")

if __name__ == "__main__":
    # Ensure the correct multiprocessing start method on Windows
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    main()
