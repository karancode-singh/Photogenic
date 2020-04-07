from flask import Flask ,render_template, jsonify
import random
import base64

# importing necessary packages
import numpy as np
import argparse
import cv2
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import time
import numpy as np
import os
import numpy as np
import pickle
import json

### OpenCV ###

def face_points_detection(img, bbox):
    PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    # return the list of (x, y)-coordinates
    return coords

from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points

def view(out):
    out = imutils.resize(out, width=1000)
    cv2.imshow("Output", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_face(bbox, im, r=10):
    points = np.asarray(face_points_detection(im, bbox))
    
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    
    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]


def smileVal(s1,s2):
    if np.isnan(s1):
        return s2

    if np.isnan(s2):
        return s1

    if s1<0.3 or s2<0.3:
        return min(s1,s2)

    return (s1+s2)/2


def swap(sF,iF,sT,iT):
        warp_2d = True ##!!
        correct_color = True ##!!
        
        # Select src face
        src_points, src_shape, src_face = select_face(sF,iF)
        # view(src_face)

        # Select dst face
        dst_points, dst_shape, dst_face = select_face(sT,iT)
        # view(dst_face)

        w, h = dst_face.shape[:2]
        
        ### Warp Image
        if not warp_2d:
                ## 3d warp
                warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (w, h))
        else:
                ## 2d warp
                src_mask = mask_from_points(src_face.shape[:2], src_points)
                src_face = apply_mask(src_face, src_mask)
                # Correct Color for 2d warp
                if correct_color:
                        warped_dst_img = warp_image_3d(dst_face, dst_points[:48], src_points[:48], src_face.shape[:2])
                        src_face = correct_colours(warped_dst_img, src_face, src_points)
                # Warp
                warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (w, h, 3))

        ## Mask for blending
        mask = mask_from_points((w, h), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask*mask_src, dtype=np.uint8)

        ## Correct color
        if not warp_2d and correct_color:
                warped_src_face = apply_mask(warped_src_face, mask)
                dst_face_masked = apply_mask(dst_face, mask)
                warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
        
        ## Shrink the mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        ##Poisson Blending
        r = cv2.boundingRect(mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

        x, y, w, h = dst_shape
        dst_img_cp = iT.copy()
        dst_img_cp[y:y+h, x:x+w] = output
        output = dst_img_cp

        return output



def getFaces(img):
    subjects = []
    fcords = []
    image = img
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            fcords.append([startX, startY, endX, endY])
            faceBoxRectangleS = dlib.rectangle(left=int(startX), top=int(startY), right=int(endX), bottom=int(endY))
            subjects.append(faceBoxRectangleS)
            # draw the bounding box of the face along with the associated
            # probability
        #     text = "{:.2f}%".format(confidence * 100)
        #     y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY),
            #             (0, 0, 255), 2)
            # cv2.putText(image, text, (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    return subjects,fcords

def eye_aspect_ratio(eye):
   A = distance.euclidean(eye[1], eye[5])
   B = distance.euclidean(eye[2], eye[4])
   C = distance.euclidean(eye[0], eye[3])
   ear = (A + B) / (2.0 * C)
   return ear

def getEAR(img,subjects):
    ears = []
    smiles = []
    image = img
    frame = image.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth= shape[mStart:mEnd]

        mouthHull = cv2.convexHull(mouth)
        
        xpoints = [x[0][0] for x in mouthHull]
        ypoints = [x[0][1] for x in mouthHull]
        
        xminidx = np.argmin(xpoints)
        y1 = ypoints[xminidx]
        x1 = xpoints[xminidx]

        yminidx = np.argmin(ypoints)
        y2 = ypoints[yminidx]
        x2 = xpoints[yminidx]

        xmaxidx = np.argmax(xpoints)
        x3 = xpoints[xmaxidx]
        y3 = ypoints[xmaxidx]

        ymaxidx = np.argmax(ypoints)
        x4 = xpoints[ymaxidx]
        y4 = ypoints[ymaxidx]

        s1 = (y1 - y2)/(x2 - x1)
        s2 = (y3 - y2)/(x3 - x2)

        smiles.append(smileVal(s1,s2))

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 2)
        ears.append(ear)

    # view(frame)

    print("smiles",smiles)
    print("ears",ears)
    return ears,smiles

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

protpath = os.path.join(os.getcwd(),'deploy.prototxt.txt')
modpath = os.path.join(os.getcwd(),'res10_300x300_ssd_iter_140000.caffemodel')

net = cv2.dnn.readNetFromCaffe(protpath, modpath)
conf = 0.36

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

def main(images):
    subs = []
    fcords = []
    eors = []
    smls = []
    numFaces = -1
    for image in images:
        s,f = getFaces(image)
        if numFaces == -1:
            numFaces = len(f)
        else:
            if not numFaces == len(f):
                print("number of faces mismatch")
                break
        s = sorted(s,key = lambda kv: kv.left())
        subs.append(s)
        fcords.append(f)
        eorsVals,smileVals = getEAR(image,s)
        eors.append(eorsVals)
        smls.append(smileVals)

    res = []
    for i in range(numFaces):
        eorsForThisFace = []
        smileForThisFace = []

        for j in range(len(images)):
            eorsForThisFace.append(eors[j][i])
            smileForThisFace.append(smls[j][i])

        earArgMax = np.argmax(eorsForThisFace)
        earMax = max(eorsForThisFace)
        if earMax < 0.2:
            res.append(earArgMax)

        else:
            for i,earVal in enumerate(eorsForThisFace):
                if earVal < 0.2:
                    smileForThisFace[i] = 10000
            res.append(np.argmin(smileForThisFace))

    print(res)
    out = images[0].copy()

    for i in range(numFaces):
        if not res[i] == 0:
            imnum = res[i]
            out = swap(subs[imnum][i],images[imnum],subs[0][i],out)

    return out

##############


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/<params>')
def strApi(params):
    print("Params")
#    print(params)
#    with open('params.pkl', 'wb') as output:
#        pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)
    imstrs = [SlashError(x) for x in params.split('*')]
#    print(imstrs)

    images = []
    for im in imstrs:
        b = base64.b64decode(im)
        with open('test.jpg','wb+') as f:
            f.write(b)
        images.append(cv2.imread('test.jpg'))
    
    print(len(images))
#    for im in images:
#        view(im)

    output = main(images)
    
#    view(output)
    cv2.imwrite('testo.jpg',output)

    return base64.b64encode(output)


def SlashError(string):
    string = string.replace('@', '/')
    string = string.replace('^', '=')
    return string

if __name__ == "__main__":
    # app.run(host = "0.0.0.0",port = "5000",debug = True)
    app.run(host='0.0.0.0')
