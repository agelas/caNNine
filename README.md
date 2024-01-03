<p align="center">
  <img src="docs/cannine.png" alt="Project Logo" width="100"/>
</p>

# caNNine
A lot of people walk some very friendly dogs past my house throughout the day. Using some inference-on-the-edge and finetuned ResNet models (the **N**eural **N**etworks in ca**NN**ine) I can be alerted to their presence and get some seratonin. 

<div align="center">
<img width="55" src="https://raw.githubusercontent.com/gilbarbara/logos/master/logos/python.svg"/>
<img width="55" src="https://raw.githubusercontent.com/gilbarbara/logos/master/logos/pytorch-icon.svg"/>
<img width="100" src="https://upload.wikimedia.org/wikipedia/commons/4/45/OpenVINO_logo.svg"/>
<img width="55" src="https://raw.githubusercontent.com/gilbarbara/logos/master/logos/flask.svg"/>
<img width="55" src="https://raw.githubusercontent.com/gilbarbara/logos/master/logos/raspberry-pi.svg"/>
</div>

# Setup
### General
The first part of the system uses a [Luxonis Oak-1 Lite](https://shop.luxonis.com/collections/oak-cameras-1/products/oak-1-lite?variant=42583148069087) camera running MobileNet SSD for detecting and tracking people and their dogs. Since the resolution isn't that great, these images are forwarded via ethernet cable to the second part of caNNine, which is a finetuned ResNet-18 model running locally, that can much more reliably categorize the dog as one worth dropping everything and going outside to pet, or not. The Luxonis camera is powered by a Raspberry Pi 4 (which in turn is powered by power over ethernet). Two small Flask apps handle requests and data transfer between the Raspberry Pi and the main computer.

### Software Setup
On the main computer, both `caNNine_server.py` and `caNNine.py` need to be running (in that order). The former serves as an intermediary server that handles requests from `caNNine.py` and forwards them to `raspberry.py`. The `caNNine.py` program acts as the primary controller that initiates and checks the status of the Raspberry Pi and the camera.

Since the Raspberry Pi is running in a headless state, it's easiest to ssh into it and start `raspberry.py` remotely. `raspberry.py` is responsible for responding to requests from `cannine_server.py` and controlling the Oak camera.
``` python
cannine.py (Main Computer)
      |
      |---(HTTP GET /check-edge)----> cannine_server.py
      |                                   |
      |<--(Response: Raspberry Pi Status)--|
      |
      |---(HTTP GET /start-oak-camera)--> cannine_server.py
      |                                   |
      |---(HTTP GET /start-camera)------> raspberry.py (Raspberry Pi)
      |                                   |
      |                                   |---(OakPipeline.run)--> Captures Image
      |                                   |
      |<--(HTTP POST /process-image)------| 
      |                                   |
      |<--(Response: Image Processing Info) 
```