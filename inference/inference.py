import torch 
import pickle 
import cv2 
import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np
from ultralytics import YOLO 
class Inference: 
    def __init__(self  ) :
        pass 

    def extract_frames(self , path) : 
        frames = []
        cap = cv2.VideoCapture(path ) 
        while True : 
            ret , frame = cap.read() 
            if not ret : 
                break 
            frames.append(frame) 
        cap.release()
        return frames 


    def track_player_and_racket(self , frames  , checkpoint , presaved = False ,saved_prediction_path = None) : 
        player_detections=[]
        if presaved and saved_prediction_path is not None:
            with open(saved_prediction_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        model = YOLO(checkpoint)
        for frame in frames : 
            
            prediction = model(frame , conf= .05  , iou = 0)[0] 
            player_detections.append(prediction) 
        if saved_prediction_path is not None:
            with open(saved_prediction_path, 'wb') as f:
                pickle.dump(player_detections, f)
        return player_detections 

    def track_ball(self, frames , checkpoint , presaved= False , saved_prediction_path = None ) : 
        ball_predictions =[]
        if presaved and saved_prediction_path is not None:
            with open(saved_prediction_path, 'rb') as f:
                ball_predictions = pickle.load(f)
            
            return ball_predictions 

        model = YOLO(checkpoint)
        for frame in frames : 
            prediction = model.predict(frame , conf =  .15)[0] 
            ball_predictions.append(prediction)
        if saved_prediction_path is not None : 
            with open(saved_prediction_path , 'wb') as f: 
                pickle.dump(ball_predictions , f)
        return ball_predictions 
    
    def court_keypoints(self , frames  , checkpoint ,presaved = False , saved_prediction_path = None ) : 
        keypoints_predictions = []
        if presaved and saved_prediction_path is not None : 
            with open(saved_prediction_path , 'rb' ) as f : 
               keypoints_predictions = pickle.load(f) 
            return keypoints_predictions
        
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 14*2) 
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for frame in frames : 
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor)
            keypoints = outputs.squeeze().cpu().numpy()
            original_h, original_w = frame.shape[:2]
            keypoints[::2] *= original_w / 224.0
            keypoints[1::2] *= original_h / 224.0
            keypoints_predictions.append(keypoints) 
        
        if saved_prediction_path is not None : 
            with open(saved_prediction_path , 'wb') as f : 
                pickle.dump(keypoints_predictions , f )
        return keypoints_predictions 


    def get_ball_missing_values_from_yolo11(self  , ball_prediction_path , player_prediction_path , collected_path = None ,collected = False ) : 
        if not collected : 
            with open(ball_prediction_path , 'rb') as f : 
                ball_predictions = pickle.load(f) 

            with open(player_prediction_path , 'rb') as f : 
                player_predictions = pickle.load(f) 
            
            new_ball_preds = [] 
            collected_missing = 0 
            for ball_pred , player_pred in zip(ball_predictions , player_predictions ) : 
                
                if  ball_pred.boxes.xyxy.numel() == 0 and 32. in player_pred.boxes.cls : 
                    new_ball_preds.append(player_pred.boxes[player_pred.boxes.cls == 32.].xyxy[0])
                    collected_missing +=1 
              
                elif ball_pred.boxes.xyxy.numel() != 0 : 
                    new_ball_preds.append(ball_pred.boxes.xyxy[0])
                else : 
                    new_ball_preds.append(ball_pred.boxes.xyxy)
            print(collected_missing) 
            with open(collected_path , 'wb') as f : 
                pickle.dump(new_ball_preds , f)
            return new_ball_preds 
        with open(collected_path , 'rb') as f : 
            new_ball_preds = pickle.load(f)
        return new_ball_preds 