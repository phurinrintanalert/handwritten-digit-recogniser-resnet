# handwritten-digit-recogniser-resnet: Handwritten Digit Recognition ✍️🔢
End-to-end handwritten digit recognition app built with a custom ResNet model trained on MNIST and deployed through Streamlit and FastAPI with interactive drawing canvas

An interactive, full-stack web application that allows users to draw a number on a digital canvas and instantly get predictions with prediction confidence from a custom-trained PyTorch ResNet model. 

## Overview

This project features a custom **ResNet architecture** trained on the MNIST dataset, deployed using  **FastAPI** backend and interactive **Streamlit** frontend. 

### Features
* **Custom ResNet Model:** ResNet optimized for 28x28 grayscale images.
* **Interactive Canvas:** A Streamlit-based drawing board using `streamlit-drawable-canvas`.
* **Preprocessing:** The backend handles RGBA transparency masking, color inversion (to match MNIST's black background), and tensor normalization.
* **Confidence Percentage:** Returns the predicted digit and model's Softmax probability/confidence score.

## How to Run Locally
Because this is a decoupled full-stack application, you need to run the backend and the frontend on separate local servers.
