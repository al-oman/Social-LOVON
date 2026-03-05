import cv2                                                                    
import numpy as np                                                            
                                                                            
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

for marker_id in [0, 1]:
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, 1000)
    cv2.imwrite(f"aruco_marker_{marker_id}.png", img)
    print(f"Saved aruco_marker_{marker_id}.png")
