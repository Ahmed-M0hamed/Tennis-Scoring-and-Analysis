from inference import Inference 
from drawing import Draw 
from homography import   draw_mini_court 
from utils import write_video 
from ball_operations import pipeline
import os 
def main():
    infernece_class = Inference() 

    frames = infernece_class.extract_frames(os.path.join(os.getcwd() , 'input_video.mp4')) 
    players_racket_predictions =infernece_class.track_player_and_racket(frames  , os.path.join(os.getcwd() ,'model_checkpoints/yolo11x.pt') , presaved = True  ,
                                                                                                saved_prediction_path = os.path.join(os.getcwd() , 'saved_preds/player_racket.pkl'))
    ball_predictions = infernece_class.track_ball(frames  , os.path.join(os.getcwd() ,'model_checkpoints/best.pt') , presaved = True ,
                                                                                           saved_prediction_path = os.path.join(os.getcwd() , 'saved_preds/ball_2.pkl'))
    court_predictions = infernece_class.court_keypoints( frames  , os.path.join(os.getcwd() ,'model_checkpoints/keypoints_model.pth') , presaved = True  , 
                                                                                           saved_prediction_path = os.path.join(os.getcwd() , 'saved_preds/court.pkl') )

    # new_ball_preds = infernece_class.get_ball_missing_values_from_yolo11(os.path.join(os.getcwd() , 'saved_preds/ball.pkl') ,os.path.join(os.getcwd() , 'saved_preds/player_racket.pkl') ,os.path.join(os.getcwd() , 'saved_preds/new_ball.pkl') , True  )
    smothed_corrected_positions , ball_court_hits , racket_hits = pipeline(os.path.join(os.getcwd() , 'saved_preds/ball_2.pkl'))
    draw_class =Draw(frames , player_racket_preds=players_racket_predictions , 
                     ball_preds=smothed_corrected_positions , court_preds= court_predictions)
    # corrected_positions = draw_class.corrected_positions(corrected=True ,
    #                                                    corrected_path=os.path.join(os.getcwd() , 'saved_preds/corrected_ball.pkl')) 
    
    print(len(ball_court_hits)) 
    print(len(racket_hits))
    draw_class.draw_court_keypoints() 
    draw_class.draw_ball_box(smothed_corrected_positions) 
    filtered_players = draw_class.draw_player_and_racket() 
    frames = draw_mini_court(frames ,court_predictions ,ball_preds= smothed_corrected_positions ,
                              player_preds= filtered_players  , ball_court_hits=ball_court_hits ,racket_hits=racket_hits )
    write_video(os.path.join(os.getcwd() , 'preds/p_7.mp4') , frames )


if __name__ == "__main__":
    main()
