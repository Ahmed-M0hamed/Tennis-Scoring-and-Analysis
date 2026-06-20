
import cv2 
import numpy as np



class KelmanFilter : 
    def __init__(self ,process_noise_pos: float = 1e-2,
    process_noise_vel: float = 5.0,
    measurement_noise: float = 1e-1 ) :
        self.kf = cv2.KalmanFilter(4, 2)   # 4 state dims, 2 measurement dims
        dt = 1.0   # one video frame = one time step

        self.kf.transitionMatrix = np.array([
        [1, 0, dt,  0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.diag([
            process_noise_pos,
            process_noise_pos,
            process_noise_vel,
            process_noise_vel,
        ]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def init_state(self, x: float, y: float) :
        """Seed the filter with the first known detection (velocity = 0)."""
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def update(self ,detection , 
    process_noise_pos: float = 1e-2,
    process_noise_vel: float = 5.0,
    measurement_noise: float = 1e-1, ) : 
        if detection is not None: 
            x , y = detection 
            self.kf.predict()
            measurement = np.array([[x], [y]], dtype=np.float32)
            self.kf.correct(measurement)
            return (x, y )
        
        predicted = self.kf.predict()          # advances the filter in time
        px = float(predicted[0, 0])
        py = float(predicted[1, 0])

        return (px, py )

    