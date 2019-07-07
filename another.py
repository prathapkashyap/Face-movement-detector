import cv2
import dlib
import numpy as np
from imutils import face_utils
#def authenticate(frame):
import time
import random
face_landmark_path = './shape_predictor_68_face_landmarks.dat'
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
questions=['turn left','turn right','tilt head up','tilt head down']
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]
def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    return reprojectdst, euler_angle

def main():
    # return
    count_right=0
    count_left=0
    count_top=0
    count_down=0
    cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Unable to connect to camera.")Qq
#         return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_rects = detector(frame, 0)
            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)
                reprojectdst, euler_angle = get_head_pose(shape)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, 	(0,0,255), -1)
                # for start, end in line_pairs:
                #     cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                
                if (int(euler_angle[1,0])>20):
                    ans='right'
                if (int(euler_angle[1,0])<-20):
                    ans='left'
                if (int(euler_angle[1,0]) in range(-20,20)):
                    ans='straight'
                if (int(euler_angle[0,0])>13):
                    ans111='down'
                if (int(euler_angle[0,0])<-17):
                    ans111='up'
                if (int(euler_angle[0,0])in range(-17,13)):
                    ans111=''
                
                
                
                
                #for right detection
                if (int(euler_angle[1,0])>20 and count_right==0):
                    now=time.time()
                    
                    cv2.imwrite('images/'+str(random.randint(0,1999999))+'.png',frame)  

                    

                    
                    count_right+=1
                    
                    
                    cv2.putText(frame, 'right detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)
                    
                if(ans=='right' and  count_right>0):
                        cv2.putText(frame, 'right already captured', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)
                        
                
                
                
                #for left detection and capture
                if (int(euler_angle[1,0])<-20 and count_left==0):
                    
                    count_left+=1
                    cv2.imwrite('images/'+str(random.randint(0,1999999))+'.png',frame)  
                    cv2.putText(frame, 'left detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)
                    
                if(ans=='left' and  count_left>0):
                    cv2.putText(frame, 'left already captured', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)


                
                #for down detection and capture
                if(int(euler_angle[0,0])>13 and count_down==0):
                    
                    count_down+=1
                    cv2.imwrite('images/'+str(random.randint(0,1999999))+'.png',frame)  
                    cv2.putText(frame, 'down detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)
                    
                if(ans111=='down' and  count_down>0):
                    cv2.putText(frame, 'down already captured', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)

                
                #for top flag detection and capture
                if (int(euler_angle[0,0])<-17 and count_top==0):
                    
                    count_top+=1
                    cv2.imwrite('images/'+str(random.randint(0,1999999))+'.png',frame)  
                    cv2.putText(frame, 'top detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)
                    
                if(ans111=='up' and  count_top>0):
                    cv2.putText(frame, 'top already captured', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)

                if(count_down==1 and count_left==1 and count_right==1 and count_top==1):
                    cv2.putText(frame,'All data requirements satisfied',(100,100),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)
                    cv2.putText(frame,'press q to exit',(120,120),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255 ), thickness=2)
                   




                    
                





                        


                
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break

            
if __name__ == '__main__':
    
    main()
