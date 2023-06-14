# importing packages

import cv2
import tensorflow as tf
import numpy as np

# set class names 

class_names = ['Incorrect_mask', 'With_mask', 'Without_mask']

# Reload the model 

model = tf.keras.models.load_model('./model-32.hdf5')

# capture the video from web camera

video = cv2.VideoCapture(0)

# select the cascade Classifier for detecting face 

faceDetect = cv2.CascadeClassifier('raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml')


while True:
    ret, frame = video.read() # read the (frame) of video and see that it right or no (ret)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting the images to grayscale

    faces = faceDetect.detectMultiScale(gray, 1.3, 3) # face detection algorithm to detect faces in the grayscale image (gray)

    for x, y, w, h in faces: #this loop through each bounding box in faces, extracts the corresponding face region from the grayscale image

        sub_face_img = gray[y : y + h, x : x + w] # cut the face region 

        resized = cv2.resize(sub_face_img, (48, 48)) # resize the face region

        normalize = resized / 255.0 # normalizes the pixel values

        reshaped = np.reshape(normalize, (1, 48, 48, 1)) # reshapes it to match the input shape of the trained model

        result = model.predict(reshaped) # Prediction result

        label = np.argmax(result, axis=1)[0] # obtain a prediction for the facial expression

        print(label)

        # set color based on predicted label
        if label == 0:
            color = (0, 255, 255)  # red for Incorrect_mask
        elif label == 1:
            color = (0, 255, 0)  # green for With_mask
        else:
            color = (0, 0, 255)  # yellow for Without_mask


        # drowing the rectangles around the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)

        # display the predicted emotion as text
        cv2.putText(frame, class_names[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    

    # displays the original color image with the bounding boxes and predicted emotion text
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# release the video and close the all windows
video.release()
cv2.destroyAllWindows()