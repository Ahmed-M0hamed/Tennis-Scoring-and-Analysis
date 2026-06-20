import numpy as np 
import cv2 
from utils import get_bottom_center_of_player , get_center_of_box , distance_between_two_points
from scoring import serving , rallying




SCALE  = 50      # pixels per metre
H_MARGIN = 300
W_MARGIN =50                  # blank border around court (px)

# Standard ITF dimensions
COURT_W         = 10.97   # doubles width
COURT_H         = 23.77   # full length
SINGLES_W       = 8.23
SERVICE_BOX_H   = 6.40
NET_Y           = COURT_H / 2


def _m2p(x_m, y_m,
         scale = SCALE, h_margin = H_MARGIN , w_margin = W_MARGIN) :
    """Convert metres → pixel (col, row)."""
    return (int(x_m * scale) + w_margin,
            int(y_m * scale) + h_margin)
def build_court_template(scale = SCALE,
                         h_margin = H_MARGIN ,HEAT = False, w_margin = W_MARGIN , BACKGROUND = (204, 37, 58)):


    W_px = int(COURT_W * scale) + 2 * w_margin
    H_px = int(COURT_H * scale) + 2 * h_margin

    img = np.zeros((H_px, W_px, 3), dtype=np.uint8)
    img[:] = BACKGROUND   # ITF green

    def mp(x, y):
        return _m2p(x, y, scale)

    WHITE = (255, 255, 255)
    T = 4

    so = (COURT_W - SINGLES_W) / 2   # singles sideline offset

    # Compute all key pixel coords
    dtl = mp(0,           0)
    dtr = mp(COURT_W,     0)
    dbr = mp(COURT_W,     COURT_H)
    dbl = mp(0,           COURT_H)
    stl = mp(so,          0)
    str_ = mp(so + SINGLES_W, 0)
    sbr = mp(so + SINGLES_W, COURT_H)
    sbl = mp(so,          COURT_H)
    net_l  = mp(0,        NET_Y)
    net_r  = mp(COURT_W,  NET_Y)
    svtl = mp(so,          SERVICE_BOX_H)
    svtr = mp(so + SINGLES_W, SERVICE_BOX_H)
    svbr = mp(so + SINGLES_W, COURT_H - SERVICE_BOX_H)
    svbl = mp(so,          COURT_H - SERVICE_BOX_H)
    svc_mid_t = mp(COURT_W / 2, SERVICE_BOX_H)
    svc_mid_b = mp(COURT_W / 2, COURT_H - SERVICE_BOX_H)

    # Draw lines
    cv2.rectangle(img, dtl, dbr, WHITE, T)          # doubles outline
    cv2.line(img, stl,  sbl,  WHITE, T)             # singles left
    cv2.line(img, str_, sbr,  WHITE, T)             # singles right
    cv2.line(img, net_l, net_r, WHITE, T + 1)       # net (thicker)
    cv2.line(img, svtl, svtr, WHITE, T)             # top service line
    cv2.line(img, svbl, svbr, WHITE, T)             # bottom service line
    cv2.line(img, svc_mid_t, svc_mid_b, WHITE, T)  # centre service line

    # Center marks on baselines
    cx = int(COURT_W / 2 * scale) + w_margin
    cv2.line(img, (cx, w_margin), (cx, w_margin + 8), WHITE, T)
    cv2.line(img, (cx, H_px - w_margin - 8), (cx, H_px - w_margin), WHITE, T)

    kp_world = np.array([
        dtl,   # 0
        dtr,   # 1
        dbr,   # 2
        dbl,   # 3
        stl,   # 4
        str_,  # 5
        sbr,   # 6
        sbl,   # 7



    ], dtype=np.float32)
    kp_list =[dtl,   # 0
        dtr,   # 1
        dbr,   # 2
        dbl,   # 3
        stl,   # 4
        str_,  # 5
        sbr,   # 6
        sbl,   # 7
        svtl ,
        svtr , 
        svbr , 
        svbl  , 
        svc_mid_t , 
        svc_mid_b , ]
    return img , kp_world , kp_list

def compute_homography(
    src_pts,
    dst_pts ,
    min_points = 4,
    ransac_thresh = 5.0
) :

    if src_pts is None or len(src_pts) < min_points:
        return None, None

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    return H, mask


def draw_minimap(
    frame_n , 
    frame,
    court_template,
    kp_frame,
    H,
    kp_world , 
    logging , 
    rally_count , 
    ball_court_hits , 
    racket_hits , 
    saved_points , 
    extra_points = None,
    position="top_right",
    TYPE = 'DYNAMIC' , 
    size_frac =  0.08,
    name = ''

) :
    
    fh, fw = frame.shape[:2]
    th, tw = court_template.shape[:2]

    target_w =   int(fw * 1.8 * size_frac) 
    target_h = int(th * int(fw  * size_frac)  / tw)
    mini = cv2.resize(court_template.copy(), (target_w, target_h))

    sx, sy = target_w / tw, target_h / th
    # Project keypoints onto mini-map

    if H is not None and kp_frame is not None:
        proj = cv2.perspectiveTransform(
            kp_frame.reshape(-1, 1, 2), H).reshape(-1, 2)
        kp = proj
        for px, py in proj:
                mx, my = int(px * sx), int(py * sy)
                if 0 <= mx < target_w and 0 <= my < target_h:
                    cv2.circle(mini, (mx, my), 3, (0, 255, 255), -1)

    # Optional extra points (players, ball, …)
    if  TYPE == 'STATIC' : 
        
        if extra_points and H is not None:
            if frame_n in ball_court_hits  : 
                for pts, color  , cls in extra_points :
                    if cls =='ball' :
                        proj_extra = cv2.perspectiveTransform(
                            pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                        saved_points['Ball'].append(proj_extra[0])
                        if len(saved_points['Ball']) == 1 :
                            call = serving(proj_extra[0] , saved_points['Player'][-1] ,kp_world )
                            logging.append(f'ball hit the ground & {call}')
                        else : 
                            call = rallying(proj_extra[0] , saved_points['Player'][-1] ,kp_world )
                            logging.append(f'ball hit the ground & {call}')
            if frame_n in racket_hits : 
                ball , player_1 , player_2 = extra_points 
                player_1_ball_distance = distance_between_two_points(ball[0][0] , player_1[0][0]) 
                player_2_ball_distance = distance_between_two_points(ball[0][0] , player_2[0][0])
                distances = [player_1_ball_distance ,player_2_ball_distance ] 
                m_index = np.argmin(distances) 
                for i ,( pts, color  , cls )in enumerate(extra_points)  :

                    if cls =='player' and i == (m_index+1) :
                        proj_extra = cv2.perspectiveTransform(
                            pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                        saved_points['Player'].append(proj_extra[0])
                        rally_count += 1 
                        logging.append(f'player {i} hit the ball & rally count = {rally_count}') 
                        
            for px, py in   saved_points['Ball']    :
                mx, my = int(px * sx), int(py * sy)
                cv2.circle(mini, (mx, my), 7, (255,0,0), -1)

            for px, py in   saved_points['Player']    :
                mx, my = int(px * sx), int(py * sy)
                cv2.putText(mini, 'X' ,(mx, my), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2 )

    if  TYPE == 'DYNAMIC' : 
        if extra_points and H is not None:
            for pts, color in extra_points:
                proj_extra = cv2.perspectiveTransform(
                    pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                for px, py in proj_extra:
                    mx, my = int(px * sx), int(py * sy)

                    cv2.circle(mini, (mx, my), 7, color  , -1)
    
    if TYPE == 'HEAT' : 
        map = np.zeros_like(mini, dtype=np.uint8)
        if extra_points and H is not None:
            for pts, color  , cls in extra_points :
                if cls =='player' :
                    proj_extra = cv2.perspectiveTransform(
                        pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                    saved_points['Players_cumilative_positions'].append(proj_extra[0])
            for px , py in saved_points['Players_cumilative_positions']  : 
                mx, my = int(px * sx), int(py * sy)
                # cv2.circle(map, (mx, my), 5, (0 ,0,255)  , -1)
                map[my , mx] += 200
            map = cv2.GaussianBlur(map, (0,0),
                    sigmaX=25,
                    sigmaY=25
                    )
            normalized = np.zeros_like(map, dtype=np.uint8)
            map = cv2.normalize(
                map,
                normalized,
                0, 255, cv2.NORM_MINMAX
            )
            map = cv2.applyColorMap(
            map,
            cv2.COLORMAP_JET ) 
            
            mini = cv2.addWeighted(mini , .5 , map , .5 , 0 )
    
    # Paste with a semi-transparent background
    pad = 80
    if position == "bottom-right":
        x0, y0 = fw - target_w - pad, fh - target_h - pad
    elif position == "bottom-left":
        x0, y0 = pad, fh - target_h - pad
    elif position == "top-right":
        x0, y0 = fw - target_w - pad, pad
    else:   # top-left
        x0, y0 = pad, pad

    roi = frame[y0:y0+target_h, x0:x0+target_w]
    frame[y0:y0+target_h, x0:x0+target_w] = cv2.addWeighted(
        roi, 0.4, mini, 0.6, 0)
    cv2.rectangle(frame,
                  (x0 - 2, y0 - 2),
                  (x0 + target_w + 2, y0 + target_h + 2),
                  (255, 255, 255), 1)
    cv2.putText(frame ,str(name) , (x0 , y0-15),cv2.FONT_HERSHEY_SIMPLEX,.9,(0, 0, 255),2,cv2.LINE_AA  )
    return frame ,saved_points , logging , rally_count 

def draw_player_info(frame , n_frame ,  distances ,prev_player_position , player_one_x ,player_one_y ,player_two_x , player_two_y ) : 
        player_one_distance = distance_between_two_points((player_one_x , player_one_y ) , prev_player_position[0]) 
        player_two_distance = distance_between_two_points((player_two_x , player_two_y ) , prev_player_position[1]) 
        distances['Player_1'].append(player_one_distance / SCALE) 
        distances['Player_2'].append(player_two_distance /SCALE )

        
        prev_player_position = ((player_one_x , player_one_y ) , (player_two_x , player_two_y ))
        velocity_1 = player_one_distance / 20
        velocity_2 = player_two_distance / 20 

        card_width , card_height = 250 , 200 
        margin = 200 
        fh, fw = frame.shape[:2]
        first_card_start = int(fw / 2 - (card_width + margin) )
        second_card_start =int( fw / 2 + margin )


        card_1 = np.zeros((card_height, card_width, 3), dtype=np.uint8)
        card_1[:] = (30,30,30)  
        cv2.putText(card_1 , f'Player 1' , (20 , int(card_height /4)) , cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA) 
        cv2.line(card_1, (first_card_start , int(card_height /4 +10)),((first_card_start+card_width ) , int(card_height /4 +10)),(80, 80, 80),1)
        cv2.putText(card_1 , f'Distance : {np.ceil(sum(distances['Player_1']))} M' , (20 , int(card_height / 2)) ,cv2.FONT_HERSHEY_SIMPLEX,0.7,(240, 240, 240),2,cv2.LINE_AA)
        cv2.putText(card_1 , f'Speed : {velocity_1:.1f} M/S' ,  (20 , int(3 * card_height / 4))  ,cv2.FONT_HERSHEY_SIMPLEX,0.7,(240, 240, 240),2,cv2.LINE_AA)  
        roi = frame[50:(50 + card_height), first_card_start:(first_card_start+card_width ) ]
        frame[50:(50 + card_height), first_card_start:(first_card_start+card_width ) ] = cv2.addWeighted(
                roi, 0.4, card_1, 0.6, 0) 
        
        card_2 = np.zeros((card_height, card_width, 3), dtype=np.uint8)
        card_2[:] = (20,20,20) 
        cv2.putText(card_2 , f'Player 2' , (20 ,int( card_height /4 )) , cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA) 
        cv2.line(card_2, (second_card_start , int(card_height /4 +10)),((second_card_start+card_width ) , int(card_height /4 +10)),(80, 80, 80),1)
        cv2.putText(card_2 , f'Distance : {np.ceil(sum(distances['Player_2']))} M' , (20 , int(card_height / 2)) ,cv2.FONT_HERSHEY_SIMPLEX,0.7,(240, 240, 240),2,cv2.LINE_AA)
        cv2.putText(card_2 , f'Speed : {velocity_2:.1f} M/S' ,  (20 , int(3 * card_height / 4))  ,cv2.FONT_HERSHEY_SIMPLEX,0.7,(240, 240, 240),2,cv2.LINE_AA)  
        roi = frame[50:(50 + card_height), second_card_start:(second_card_start+card_width ) ]
        frame[50:(50 + card_height), second_card_start:(second_card_start+card_width ) ] = cv2.addWeighted(
                roi, 0.4, card_2, 0.6, 0) 
            
        cv2.rectangle(frame,
                  ( first_card_start - 2 ,50 - 2),
                  ( first_card_start + card_width + 2 ,50 + card_height + 2),
                  (255, 255, 255), 1)
        cv2.rectangle(frame,
                  ( second_card_start - 2,50 - 2),
                  ( second_card_start + card_width + 2,50 + card_height + 2),
                  (255, 255, 255), 1)

        return frame , distances , prev_player_position 
def draw_logging(frame , frame_n , logs  ) : 
    card_width , card_height = 650 , 200 
    margin = 20 
    side_margin = 80
    fh, fw = frame.shape[:2]
    log_card_start = int(fh - card_height - 250 )  



    log_card = np.zeros((card_height, card_width, 3), dtype=np.uint8)
    log_card[:] = (60,60,60)   
    cv2.putText(frame , str("REF") , ( side_margin,log_card_start -10) ,cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 0, 255),4,cv2.LINE_AA)
    for i , log in enumerate(logs[-3:]) : 
        if log.split()[-1] == 'inside' : 
            color = (0 ,255 ,0)
        elif log.split()[-1] == 'outside' : 
            color = (0 ,0 ,255)
        else : color = (255 ,255,255)
        cv2.putText(log_card , str(log) , (margin  , int((i+1) * card_height / 4 )) , cv2.FONT_HERSHEY_SIMPLEX,1.0,color,2,cv2.LINE_AA) 
    roi = frame[log_card_start:(log_card_start + card_height), side_margin:(side_margin + card_width ) ]
    frame[log_card_start:(log_card_start + card_height), side_margin:(side_margin + card_width ) ] = cv2.addWeighted(
                roi, 0.4, log_card, 0.6, 0)
    cv2.rectangle(frame,
                  (side_margin - 2, log_card_start - 2),
                  (side_margin + card_width + 2, log_card_start + card_height + 2),
                  (255, 255, 255), 1) 
    return frame 
def draw_mini_court(
    frames , 
    court_preds , 
    ball_preds , 
    player_preds , 
    ball_court_hits , 
    racket_hits , 


) :

    court_template_1, kp_world  , kp_list  = build_court_template()
    court_template_2, kp_world , kp_list = build_court_template(BACKGROUND=(0,0,0))
    head_tempelete  , kp_world  , kp_list=  build_court_template( )



    saved_points = {'Ball' : [] , 'Player' : [] , 'Players_cumilative_positions' : []} 
    distances = {'Player_1' : [] , 'Player_2' : []} 
    rally_count = 0 
    logging = []
    prev_player_position = (get_bottom_center_of_player(player_preds[0][0]) , get_bottom_center_of_player(player_preds[0][1]))
    for i , (frame , court_pred , ball_pred , player_pred ) in enumerate (zip(frames , court_preds , ball_preds , player_preds) ) :

        # ── (a) model inference ──────────────────────────────
        x0 ,y0 , x1 ,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13 = court_pred
        kp_frame = np.array([(x0 ,y0 ) ,( x1 ,y1)
                              ,(x3,y3),(x2,y2)
                              ,(x4,y4),(x6,y6),
                              (x7,y7),(x5,y5)
                             ])


        
        H, mask = compute_homography(kp_frame, kp_world)
           # too stale, give up
        x_ball , y_ball = ball_pred 
        # print(f'player_pred {player_pred[0]} ')
        player_one_x , player_one_y = get_bottom_center_of_player(player_pred[0])
        player_two_x , player_two_y = get_bottom_center_of_player(player_pred[1])

        


        # ── (c) overlays ─────────────────────────────────────
        if H is not None:

            


            frame , _ , logging , rally_count= draw_minimap(
                    i , 
                    frame, court_template_1,
                    kp_frame, H,
                    position='top-left',
                    extra_points=[ (np.array([(x_ball, y_ball)]) , (255,0,0) )
                                   ,(np.array([(player_one_x, player_one_y)]) , (0,0,255) )  ,
                                   (np.array([(player_two_x, player_two_y)]) , (0,0,255) )] , 
                    kp_world =kp_list , 
                    ball_court_hits= ball_court_hits , 
                    TYPE= 'DYNAMIC' , 
                    racket_hits= racket_hits  , 
                    saved_points=saved_points , 
                    logging = logging , rally_count = rally_count , name='Ball-Player Positions'
                )
            
            ball_court_hits.sort() 
            racket_hits.sort()


            frame  , saved_points  , logging , rally_count = draw_minimap(
                    i , 
                    frame, court_template_2,
                    kp_frame, H,
                    position='top-right',
                    extra_points=[ (np.array([(x_ball, y_ball)]) , (255,0,0) , 'ball')
                                   ,(np.array([(player_one_x, player_one_y)]) , (0,0,255) ,'player')  ,
                                   (np.array([(player_two_x, player_two_y)]) , (0,0,255) , 'player')] , 
                    kp_world =kp_list , 
                    TYPE= 'STATIC'  , 
                    ball_court_hits= ball_court_hits , 
                    racket_hits= racket_hits  , 
                    saved_points = saved_points , logging = logging , rally_count = rally_count , name = 'Ball-Player Footprint'
                )
            frame  , saved_points  , logging  , rally_count= draw_minimap(
                    i , 
                    frame, head_tempelete,
                    kp_frame, H,
                    position='bottom-right',
                    extra_points=[ (np.array([(x_ball, y_ball)]) , (255,0,0) , 'ball')
                                   ,(np.array([(player_one_x, player_one_y)]) , (0,0,255) ,'player')  ,
                                   (np.array([(player_two_x, player_two_y)]) , (0,0,255) , 'player')] , 
                    kp_world =kp_list , 
                    TYPE= 'HEAT'  , 
                    ball_court_hits= ball_court_hits , 
                    racket_hits= racket_hits  , 
                    saved_points = saved_points , logging = logging , rally_count = rally_count , name ='Heat Map'
                )
            frame , distances , prev_player_position  = draw_player_info(frame = frame , n_frame= i , prev_player_position=prev_player_position
                                                                         , distances= distances , player_one_x=player_one_x , player_one_y= player_one_y
                                                                         , player_two_x= player_two_x , player_two_y = player_two_y)
            frame = draw_logging(frame , i , logging)
    return frames 

