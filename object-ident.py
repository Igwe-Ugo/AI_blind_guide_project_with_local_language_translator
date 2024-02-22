import os
import cv2
import json
import time
import numpy as np
import tkinter as tk
import urllib.request
from PIL import Image, ImageTk

# Initialize variables
classNames = []  # List to store class names
classFile = 'coco.names' # Path to the file containing class names
audio_json = 'object_sound.json'

#thres = 0.45 # Threshold to detect object
video_url = "http://192.168.0.101/cam-hi.jpg" # gets the video url from the ESP-32 CAM
cap = cv2.VideoCapture(video_url) # Captures the video
confirmed = False # Flag to check if the confirmation sound has been played

# Create a Tkinter root window
root = tk.Tk()
root.wm_title("Video Capture") # Set window title
imageFrame = tk.Frame(root, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)
lmain = tk.Label(imageFrame) # Create a label to display the video frame
lmain.grid(row=0, column=0)

# Function to play confirmation sound
def confirmVideoCapture():
    global confirmed
    # Check if the IP camera stream is opened successfully
    if not cap.isOpened():
        try:
            if not confirmed:
                root.geometry('500x300')
                label = tk.Label(root, text="Enweghi mgbaama vidiyo enwetara\nsite na ugogbe anya!", font=('Courier', 16, 'bold'))
                label.place(x=30, y=90)
                os.system("mpg321 audio/camera-off.mp3 &")
                confirmed = True
        except HTTPError or URLError:
            if not confirmed:
                root.geometry('500x300')
                label = tk.Label(root, text="Enweghi mgbaama vidiyo enwetara\nsite na ugogbe anya!", font=('Courier', 16, 'bold'))
                label.place(x=30, y=90)
                os.system("mpg321 audio/camera-off.mp3 &")
                confirmed = True
    else:
        if not confirmed:
            os.system("mpg321 audio/camera-on.mp3 &")
            confirmed = True

# Open the file containing class names and store them in a list. 
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load the configuration and weights files for object detection
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

# Configure the  object detection model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to detect objects in the frame
def getObjects(img, thres, nms, draw=True, objects=[]):
    # Detect objects in the frame
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo =[] # List to store object information
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    # Draw the bounding box around detected object
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img,objectInfo

# Function to find audio file corresponding to detected object
def findSound(classIdentifier):
    # returns the audio path
    try:
        with open(audio_json) as f:
            objData = json.load(f)
            for entry in objData:
                if entry.get("item") == classIdentifier:
                    ret = entry.get("audio")
                    return ret
            return None
    except FileNotFoundError:
        return None

# Function to update the video frmane and detect objects
def update_frame():
    confirmVideoCapture() # Play confirmation sound function
    img_resp = urllib.request.urlopen(video_url) # request for the url to be open
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8) # converts the image url to an array/list
    image = cv2.imdecode(imgnp, -1) # decodes the array and picks the last value in the list/array
    frame = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Detect objects in the frame
    _, objectInfo = getObjects(frame,0.45,0.2)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Create Tkinter-compatible image from frame
    img = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    # Store img_tk as an attribute of label
    lmain.img_tk = img_tk
    lmain.configure(image=img_tk)

    # play audio corresponding to detected objects
    for item in objectInfo:
        objectId = item[-1]
        sound = findSound(objectId)
        os.system(f"mpg321 {sound} &")
        time.sleep(3) # Add a delay of 3 seconds between playong each audio file

    # Schedule the update_frame function to be called after 10 milliseconds
    lmain.after(10, update_frame)

# Main function
if __name__ == "__main__":
    update_frame() # Start the update_frame loop
    root.mainloop() # Start the Tkinter event loop
