from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.cams import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import os
from PIL import Image
import torch
from medmnist.models import ResNet18, ResNet50

path ='C:\\code\\faster-rcnn-pytorch-master\\ckpt_97_auc_0.99310.pth'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(in_channels=3,num_classes=2)# 定义model

state_dict = torch.load(path, map_location=device)
model.load_state_dict(state_dict,strict=False)
cam_extractor = SmoothGradCAMpp(model)   # 将model载入CAM中
# Get your input
img1 = read_image("./1.jpg")
# Preprocess it for your chosen model
img = torch.cat((img1,img1))
img = torch.cat((img,img1))
print(img.shape)
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 对图片进行处理
# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map, mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
