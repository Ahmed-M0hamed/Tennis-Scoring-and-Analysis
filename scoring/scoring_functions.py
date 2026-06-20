import numpy as np 
from utils import distance_between_two_points 
def serving(ball_position , last_racket_hit_position , kp ): 
    # kp_world = np.array([
    #     dtl,   # 0
    #     dtr,   # 1
    #     dbr,   # 2
    #     dbl,   # 3
    #     stl,   # 4
    #     str_,  # 5
    #     sbr,   # 6
    #     sbl,   # 7
    #     svtl ,
    #     svtr , 
    #     svbr , 
    #     svbl  , 
    #     svc_mid_t , 
    #     svc_mid_b , 


    # ], dtype=np.float32) 
    distances = []
    for point in kp[:4] : 
        d = distance_between_two_points(last_racket_hit_position , point ) 
        distances.append(d) 
    call = None 
    min_distance = np.argmin(distances) 
    if min_distance == 0 : 
        bx , by = ball_position 
        if bx > kp[-2][0] and bx < kp[9][0] and by < kp[-1][1] and by > (kp[-2][1]+kp[-1][1]) / 2 : 
            call = 'serve inside' 
        else : 
            call = 'seve outside' 
    if min_distance == 1 : 
        bx , by = ball_position 
        if bx > kp[8][0] and bx < kp[-2][0] and by < kp[-1][1] and by > (kp[-2][1]+kp[-1][1]) / 2 : 
            call = 'serve inside' 
        else : 
            call = 'seve outside'  
    if min_distance == 2 : 
        bx , by = ball_position 
        if bx > kp[8][0] and bx < kp[-2][0] and by > kp[-1][1] and by < (kp[-2][1]+kp[-1][1]) / 2 : 
            call = 'serve inside' 
        else : 
            call = 'seve outside'  
    if min_distance == 3 : 
        bx , by = ball_position 
        if bx > kp[-2][0] and bx < kp[9][0] and by > kp[-1][1] and by < (kp[-2][1]+kp[-1][1]) / 2 : 
            call = 'serve inside' 
        else : 
            call = 'serve outside'  
    return call 


def rallying(ball_position , last_racket_hit_position , kp) : 
    distances = []
    for point in kp[:4] : 
        d = distance_between_two_points(last_racket_hit_position , point ) 
        distances.append(d) 
    
    call = None 
    min_distance = np.argmin(distances) 
    if min_distance == 0 or min_distance == 1 : 
        bx , by = ball_position 
        if bx > kp[7][0] and bx < kp[6][0] and by < kp[6][1] and by > (kp[-2][1]+kp[-1][1]) / 2 : 
            call = 'ball inside' 
        else : 
            call = 'ball outside' 
    if min_distance == 2 or min_distance == 2 : 
        bx , by = ball_position 
        if bx > kp[4][0] and bx < kp[5][0] and by < kp[4][1] and by < (kp[-2][1]+kp[-1][1]) / 2 : 
            call = 'ball inside' 
        else : 
            call = 'ball outside' 
    return call 