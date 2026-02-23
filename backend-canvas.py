from fastapi import FastAPI, UploadFile,File
from PIL import Image, ImageOps
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from prediction_model_new import my_ResNet4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
loaded_resnet = my_ResNet4(in_channels =1 )
loaded_resnet.load_state_dict(torch.load("mnist_model.pth",weights_only = True,map_location = device))
loaded_resnet.to(device)
loaded_resnet.eval()

app = FastAPI()
@app.post('/canvas/')
async def process_image(image_sent :UploadFile = File(...)):
    contents = await image_sent.read()
    img = Image.open(io.BytesIO(contents))
    if img.mode == 'RGBA':
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        image = background.convert('L')
    else:
        image = Image.open(io.BytesIO(contents))

    image_inverted = ImageOps.invert(image)
    preprocess = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    image_tensor = preprocess(image_inverted)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = loaded_resnet(image_tensor)
        probabilities = F.softmax(output, dim=1)
        _,predicted_number = torch.max(output,1)
    return {"predicted_number":predicted_number.item(),
            "probabilities":probabilities[0][predicted_number].item()*100,
            "message": "Image processed successfully"}