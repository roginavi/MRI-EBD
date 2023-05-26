# Author Ivan Igor (ivanigorg@proton.me)
import cv2
import imutils
from imutils import face_utils
import csv
import numpy as np
import dlib
from scipy.spatial import distance as dist

# functions
def get_EAR(points):
    return (dist.euclidean(points[1], points[5]) + dist.euclidean(points[2],points[4])) / (2.0 * dist.euclidean(points[0], points[3]))

# load predictor
predictor = dlib.shape_predictor('./model/eye_model.dat')

# load video
filename = 'ishare_short.mp4'
video = cv2.VideoCapture('./' + filename)
if (video.isOpened() == False):
    print('Error opening video file')
n = video.get(cv2.CAP_PROP_FRAME_COUNT)
print(n)
middle_frame = np.round(n/2)
video.set(cv2.CAP_PROP_POS_FRAMES,middle_frame)

# select eyes from middle frame
ret, frame = video.read()
image = imutils.resize(frame, width = 400)
gray_f = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
r = cv2.selectROI("select",gray_f)
video.set(cv2.CAP_PROP_POS_FRAMES,0)
# create csv file
header = ['frame','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6','EAR']
f = open('./test.csv','w')
writer = csv.writer(f)
writer.writerow(header)

while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        image = imutils.resize(frame, width = 400)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.rectangle(image, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0),1)
        rect = dlib.rectangle(r[0],r[1],r[0]+r[2],r[1]+r[3])
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        n_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        writer.writerow(np.hstack((n_frame, shape.flatten(), get_EAR(shape))))
        for (x,y) in shape:
            cv2.circle(image,(x,y),1,(0,0,255),-1)
        cv2.imshow(filename, image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
f.close()
video.release()
cv2.destroyAllWindows()



