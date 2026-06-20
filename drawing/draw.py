import cv2
import numpy as np 
import pickle
from utils import distance_between_two_points , get_bottom_center_of_player  , get_center_of_box
from players_clustring_and_information import cluster_players 
from utils import KelmanFilter
class Draw : 
    def __init__(self ,frames,player_racket_preds , ball_preds , court_preds ) : 
        self.frames = frames 
        self.player_racket_preds = player_racket_preds 
        self.ball_preds = ball_preds 
        self.court_preds = court_preds 


    def write_video(self ,  output_path) : 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 24, (self.frames[0].shape[1], self.frames[0].shape[0]))
        for frame in self.frames :
            out.write(frame)
        out.release()
    def filter_persons(self , player_racket_pred , two_points ) : 
        distances = {'point_12' : [] , 'point_13' : []}
        filtred_players = []
        for point_num , point in two_points.items() :
            for person in player_racket_pred.boxes[player_racket_pred.boxes.cls == 0.].xyxy :
                x_center , y_center = get_bottom_center_of_player(person) 
                distance = distance_between_two_points((x_center , y_center) , point) 
                distances[point_num].append(distance) 
                
        player_1_top_screen =  player_racket_pred.boxes[player_racket_pred.boxes.cls == 0.].xyxy[np.argmin(distances['point_12'])]
        filtred_players.append(player_1_top_screen)
        player_2_bottom_screen = player_racket_pred.boxes[player_racket_pred.boxes.cls == 0.].xyxy[np.argmin(distances['point_13'])]
        filtred_players.append(player_2_bottom_screen )

        return filtred_players  
    def draw_court_keypoints(self )  : 
        pairs = [(1,6) , (4,6) , (6,9) , (0,4) , (4,8) , (0,2) , (9,12) , (1,3) , (8,12) , 
                 (3,7) , (5,7) , (2,5) , (10,13) , (7,11) , (11,13) , (5,10) , (8,10) ,(9,11), (12,13)] 
        for keypoints_pred , frame in zip(self.court_preds , self.frames) : 
            for i  in range(0 ,len(keypoints_pred) , 2) : 
                x = int(keypoints_pred[i] ) 
                y = int(keypoints_pred[i+1])
                cv2.circle(frame , (x, y) , 5, (0, 0, 255), -1)
                cv2.putText(frame , str(f'point_{i//2}') , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for pair in pairs : 
                point_1 , point_2 = pair 
                x1 , y1 = int(keypoints_pred[point_1 * 2]) , int(keypoints_pred[point_1 * 2 +1])
                x2, y2 = int(keypoints_pred[point_2 *2] ) , int(keypoints_pred[point_2 * 2 +1])
                cv2.line(frame , (x1 , y1 ) , (x2,y2) ,(0, 0, 255) , 2 )

    
        
    def draw_ball_box(self , corrected_postions) : 
        for i , (ball_pred , frame) in enumerate(zip(corrected_postions, self.frames )) :
        
                x, y = ball_pred 
                cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
                
                    
    def draw_player_and_racket(self ) : 
        filtered_players_all = []
        for player_racket_pred , frame , court_pred in zip(self.player_racket_preds , self.frames , self.court_preds) : 
            if player_racket_pred.boxes[player_racket_pred.boxes.cls == 38.].xyxy.numel() != 0 : 
                for racket in player_racket_pred.boxes[player_racket_pred.boxes.cls == 38.].xyxy : 
                    x1 , y1 , x2 , y2 = racket 
                    cv2.rectangle(frame, (int(x1) , int(y1) ),( int(x2) , int(y2)) ,(0, 255, 255), 2) 

            two_points = {'point_12' : (court_pred[24] , court_pred[25]) , 'point_13' : (court_pred[26] , court_pred[27]) }
            filtered_players = self.filter_persons(player_racket_pred ,two_points)       
            filtered_players_all.append(filtered_players) 
            for person in filtered_players : 
                x1 , y1 , x2 , y2 = person 

                cv2.rectangle(frame, (int(x1) , int(y1) ) ,(int(x2) , int(y2)) ,(0, 255, 255), 2) 

        return filtered_players_all