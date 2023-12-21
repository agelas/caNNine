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
        self.pipeline = self.create_pipeline()

    def create_pipeline(self):
        labelMap = ["person", ""]

        nnPathDefault = str((Path(__file__).parent / Path('./person-detection-retail-0013_openvino_2021.4_7shave.blob')).resolve().absolute())
        parser = argparse.ArgumentParser()
        parser.add_argument('-nnPath', help="Path to mobilenet detection network blob", default=nnPathDefault)

        args = parser.parse_args()

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        manip = pipeline.create(dai.node.ImageManip)
        objectTracker = pipeline.create(dai.node.ObjectTracker)
        detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)

        manipOut = pipeline.create(dai.node.XLinkOut)
        xinFrame = pipeline.create(dai.node.XLinkIn)
        trackerOut = pipeline.create(dai.node.XLinkOut)
        xlinkOut = pipeline.create(dai.node.XLinkOut)
        nnOut = pipeline.create(dai.node.XLinkOut)

        manipOut.setStreamName("manip")
        xinFrame.setStreamName("inFrame")
        xlinkOut.setStreamName("trackerFrame")
        trackerOut.setStreamName("tracklets")
        nnOut.setStreamName("nn")

        # Properties
        xinFrame.setMaxDataSize(1920*1080*3)

        manip.initialConfig.setResizeThumbnail(544, 320)
        # manip.initialConfig.setResize(384, 384)
        # manip.initialConfig.setKeepAspectRatio(False) #squash the image to not lose FOV
        # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip.inputImage.setBlocking(True)

        # setting node configs
        detectionNetwork.setBlobPath(args.nnPath)
        detectionNetwork.setConfidenceThreshold(0.5)
        detectionNetwork.input.setBlocking(True)

        objectTracker.inputTrackerFrame.setBlocking(True)
        objectTracker.inputDetectionFrame.setBlocking(True)
        objectTracker.inputDetections.setBlocking(True)
        objectTracker.setDetectionLabelsToTrack([1])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        manip.out.link(manipOut.input)
        manip.out.link(detectionNetwork.input)
        xinFrame.out.link(manip.inputImage)
        xinFrame.out.link(objectTracker.inputTrackerFrame)
        detectionNetwork.out.link(nnOut.input)
        detectionNetwork.out.link(objectTracker.inputDetections)
        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        objectTracker.out.link(trackerOut.input)
        objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

        return pipeline

    def run(self):
        pass

    def send_image(self, image):
        response = requests.post(f'{self.host_url}/process-image', headers=self.headers, files={'image': image})
        return response.content

