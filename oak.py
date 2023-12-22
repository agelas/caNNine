import argparse
from pathlib import Path
import cv2
import depthai as dai
import numpy as np 
import time
import requests 

class OakPipeline:
    def __init__(self, host_url, api_key):
        self.host_url = host_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.pipeline: dai.Pipeline = self.create_pipeline()
        self.label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def create_pipeline(self) -> dai.Pipeline:
        nnPathDefault = str((Path(__file__).parent / Path('./mobilenet/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
        parser = argparse.ArgumentParser()
        parser.add_argument('-nnPath', help="Path to mobilenet detection network blob", default=nnPathDefault)
        parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

        args = parser.parse_args()

        fullFrameTracking = args.full_frame

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        objectTracker = pipeline.create(dai.node.ObjectTracker)

        xlinkOut = pipeline.create(dai.node.XLinkOut)
        trackerOut = pipeline.create(dai.node.XLinkOut)

        xlinkOut.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

        # testing MobileNet DetectionNetwork
        detectionNetwork.setBlobPath(args.nnPath)
        detectionNetwork.setConfidenceThreshold(0.5)
        detectionNetwork.input.setBlocking(False)

        objectTracker.setDetectionLabelsToTrack([12, 15])
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        camRgb.preview.link(detectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

        if fullFrameTracking:
            camRgb.video.link(objectTracker.inputTrackerFrame)
        else:
            detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        detectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.out.link(trackerOut.input)

        return pipeline

    def run(self): 
        with dai.Device(self.pipeline) as device:
            preview = device.getOutputQueue("preview", 4, False)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            frame = None

            while(True):
                imgFrame = preview.get()
                track = tracklets.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                color = (255, 0, 0)
                frame = imgFrame.getCvFrame()
                trackletsData = track.tracklets
                if(trackletsData.len() > 1):
                    self.send_image(frame)
                for t in trackletsData:
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    try:
                        label = self.labelMap[t.label]
                    except:
                        label = t.label

                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                cv2.imshow("tracker", frame)

                if cv2.waitKey(1) == ord('q'):
                    break
        
    
    def send_image(self, image):
        retval, buffer = cv2.imencode('.jpg', image)
        if retval:
            image_bytes = np.array(buffer).tobytes()
            response = requests.post(f'{self.host_url}/process-image', headers=self.headers, files={'image': ('image.jpg', image_bytes, 'image/jpeg')})
            return response.content
        else:
            print("Failed to encode image")
            return None

