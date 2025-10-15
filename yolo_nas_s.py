# import os
# import multiprocessing as mp

# # Set the environment variable to allow duplicate OpenMP runtime initialization
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from ultralytics import YOLO

# def main():
#     os.chdir(r'D:\aanya')

#     # Load your custom dataset
#     # your_dataset = r'D:\aanya\crack_data\data.yaml'

#     # Initialize the YOLO-NAS model
#     model = YOLO('yolo_nas_s.pt')  # Load YOLO-NAS with small variant

#     # Define your training parameters
#     train_params = {
#         'data': r'D:\aanya\crack_data\data.yaml',
#         'epochs': 10,
#         'batch': 16,
#         'lr': 0.001,
#         'imgsz': 640,
#         'device': '0'
#     }

#     # Train the model on your dataset
#     model.train(**train_params)
#     print("Training complete.")

# if __name__ == "__main__":
#     # Required on Windows to prevent multiprocessing errors
#     if mp.get_start_method(allow_none=True) != 'spawn':
#         mp.set_start_method('spawn', force=True)
    
#     main()
# from ultralytics import NAS
# import torch
# if __name__ == '__main__':
#     torch.use_deterministic_algorithms(True, warn_only=True)
#     model = NAS(r'D:\aanya\crack_data\yolo_nas_s.pt')
#     results = model.train(data=r'D:\aanya\crack_data\data.yaml', epochs=5, imgsz=640, batch=8)

import super_gradients

yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="").cuda()
model_predictions  = yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg").show()

# prediction = model_predictions[0].prediction        # One prediction per image - Here we work with 1 image so we get the first.

# bboxes = prediction.bboxes_xyxy                     # [[Xmin,Ymin,Xmax,Ymax],..] list of all annotation(s) for detected object(s) 
# bboxes = prediction.bboxes_xyxy                     # [[Xmin,Ymin,Xmax,Ymax],..] list of all annotation(s) for detected object(s) 
# class_names = prediction.class_names                # ['Class1', 'Class2', ...] List of the class names
# class_name_indexes = prediction.labels.astype(int)  # [2, 3, 1, 1, 2, ....] Index of each detected object in class_names(corresponding to each bounding box)
# confidences =  prediction.confidence.astype(float)  # [0.3, 0.1, 0.9, ...] Confidence value(s) in float for each bounding boxes