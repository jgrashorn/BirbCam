import numpy as np
import cv2
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import torchvision
import cv2
import imutils
import threading

# Replace with your Raspberry Pi's IP and streaming port
udp_stream_url = 'udp://@192.168.178.83:8081?pkt_size=1316'
latest_frame = None

def capture_stream():
    global latest_frame
    cap = cv2.VideoCapture(udp_stream_url)
    if not cap.isOpened():
        print("Error: Could not open UDP stream.")
        return
    
    while True:
        # Capture the latest frame and update the shared variable
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        else:
            print("Warning: No frame received.")
            break

    cap.release()

def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
        choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
        help="name of the object detection model")
    ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
        help="path to file containing list of categories in COCO dataset")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # set the device we will be using to run the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    # load the list of categories in the COCO dataset and then generate a
    # set of bounding box colors for each class
    CLASSES = pickle.loads(open(args["labels"], "rb").read())
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # initialize a dictionary containing model name and its corresponding 
    # torchvision function call
    MODELS = {
        "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
        "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        "retinanet": detection.retinanet_resnet50_fpn
    }
    # load the model and set it to evaluation mode
    model = MODELS[args["model"]](
                                    weights=detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                    progress=True,
                                    num_classes=len(CLASSES),
                                    weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
                                ).to(DEVICE)
    
    model.eval()

    # Stream
    # ______________________________________________________________

    # Start the capture thread
    capture_thread = threading.Thread(target=capture_stream)
    capture_thread.daemon = True
    capture_thread.start()

    video = cv2.VideoWriter('video.mp4',-1,10,(1920,1080))

    # Capture frames from the UDP stream
    while True:
        # Read frame-by-frame
        if latest_frame is not None:
            frame = latest_frame.copy()
            # load the image from disk
            # image = imutils.resize(frame, width=400)
            image = frame
            orig = image.copy()
            # convert the image from BGR to RGB channel ordering and change the
            # image from channels last to channels first ordering
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            # add the batch dimension, scale the raw pixel intensities to the
            # range [0, 1], and convert the image to a floating point tensor
            image = np.expand_dims(image, axis=0)
            image = image / 255.0
            image = torch.FloatTensor(image)
            # send the input to the device and pass the it through the network to
            # get the detections and predictions
            image = image.to(DEVICE)
            detections = model(image)[0]

            # loop over the detections
            for i in range(0, len(detections["boxes"])):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections["scores"][i]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                    # extract the index of the class label from the detections,
                    # then compute the (x, y)-coordinates of the bounding box
                    # for the object
                    idx = int(detections["labels"][i])
                    box = detections["boxes"][i].detach().cpu().numpy()
                    (startX, startY, endX, endY) = box.astype("int")
                    # display the prediction to our terminal
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("[INFO] {}".format(label))
                    # draw the bounding box and label on the image
                    cv2.rectangle(orig, (startX, startY), (endX, endY),
                        COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(orig, label, (startX, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, COLORS[idx], 1)
            # show the output image

            # Display the frame (optional)
            # image = imutils.resize(orig, width=400)
            cv2.imshow('UDP Stream', orig)
            video.write(orig)

            # Process the frame for machine vision here
            # ...

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and close any OpenCV windows
    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()