import numpy as np 
import pandas as pd 
import pickle 
from .smothing import smooth_trajectory 
from scipy.signal import savgol_filter, find_peaks
def prepare_df(path ):
    with open(path, 'rb') as f:
        ball_positions = pickle.load(f)
    ball_positions = [x.boxes.xyxy[0].numpy() if x.boxes.xyxy.numel() != 0 else []  for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
    df_ball_positions['frame'] = df_ball_positions.index 
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()
    df_ball_positions['cx'] = (df_ball_positions['x1'] + df_ball_positions['x2' ]) /2 
    df_ball_positions['cy'] = (df_ball_positions['y1'] + df_ball_positions['y2']) /2 
    new_df =  df_ball_positions[['frame' , 'cx' , 'cy'] ] 
    smothed_df = smooth_trajectory(new_df ) 
    smothed_df['mid_y_rolling_mean'] = smothed_df['cy_smooth'].rolling(window=5, min_periods=1, center=False).mean()
    smothed_df['delta_y'] = smothed_df['mid_y_rolling_mean'].diff()
    return smothed_df 

def detect_bounces(
    df,min_drop_px = 10.0, min_gap_frames= 15 , court_y_min = 1,   # ignore bounces above this y (pixels)
) :
    cy = df["delta_y"].values
    # find_peaks on cy detects local maxima (lowest on screen = bounce)
    peaks, p_props = find_peaks(
        cy,
        height=court_y_min,           # optional lower bound on y
        prominence=min_drop_px,       # must stick out by N px
        distance=min_gap_frames,      # minimum spacing
    )
    y_min = -1 * court_y_min if court_y_min is not None else None 
    bottoms , b_props= find_peaks(
        -1 * cy,
        height= y_min,           # optional lower bound on y
        prominence=min_drop_px,       # must stick out by N px
        distance=min_gap_frames,      # minimum spacing
    )

    peaks_frames = df["frame"].iloc[peaks].values
    bottoms_frames = df["frame"].iloc[bottoms].values
    bounce_frames = np.concatenate((peaks_frames, bottoms_frames))
    props = np.concatenate((p_props['prominences'] ,b_props['prominences'] ))

    bounce_df = pd.DataFrame({
        "bounce_frame": bounce_frames , 
        "cy_at_bounce": cy[bounce_frames],
        "cx_at_bounce": df["cx_smooth"].values[bounce_frames],
        "prominence_px": props,
    })
    return bounce_df 

def detect_racket_hits(df) : 

    df['ball_hit']=0
    minimum_change_frames_for_hit = 25
    for i in range(1,len(df)- int(minimum_change_frames_for_hit*1.2) ):
        negative_position_change = df['delta_y'].iloc[i] >0 and df['delta_y'].iloc[i+1] <0
        positive_position_change = df['delta_y'].iloc[i] <0 and df['delta_y'].iloc[i+1] >0

        if negative_position_change or positive_position_change:
            change_count = 0 
            for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                negative_position_change_following_frame = df['delta_y'].iloc[i] >0 and df['delta_y'].iloc[change_frame] <0
                positive_position_change_following_frame = df['delta_y'].iloc[i] <0 and df['delta_y'].iloc[change_frame] >0

                if negative_position_change and negative_position_change_following_frame:
                    change_count+=1
                elif positive_position_change and positive_position_change_following_frame:
                    change_count+=1
        
            if change_count>minimum_change_frames_for_hit-1:
                df.loc[df.index[i], 'ball_hit'] = 1

    frame_nums_with_ball_hits = df[df['ball_hit']==1].index.tolist()
    
    return frame_nums_with_ball_hits

def find_closest(lst, target):
    return min(lst, key=lambda x: abs(x - target))

def pipeline(path ) : 
    smothed_df = prepare_df(path) 
    bounce_df = detect_bounces(smothed_df ,min_drop_px=5,
    min_gap_frames=20,
    court_y_min=1)
    racket_hits_df = detect_racket_hits(smothed_df) 

    racket_hits_detections = racket_hits_df
    bounce_detections  = bounce_df.bounce_frame.values.tolist()

    racket_hits = []
    for v in racket_hits_detections : 
        closest = find_closest(bounce_detections , v)  

        racket_hits.append(int((v+closest) /  2 )) 
        bounce_detections.remove(closest)
    
    ball_court_hits = bounce_detections.copy() 
    positions  = smothed_df[['cx_smooth' ,'cy_smooth']].to_numpy()

    return positions ,ball_court_hits, racket_hits