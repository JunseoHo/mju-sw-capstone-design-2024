from super_gradients.training import models
from tqdm.auto import tqdm
import torch
import os

test_imgs_dir = '../test_dataset/images/'
test_imgs = os.listdir(test_imgs_dir)

model_name = 'yolo_nas_s'
checkpoint_path = '../checkpoints/yolo_nas_s/RUN_20240601_204014_994191/ckpt_best.pth'
classes = ['sp', 'cn', 'la', 'tr', 'lp']

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = models.get(
    model_name=model_name,
    checkpoint_path=checkpoint_path,
    num_classes=len(classes)
).to(device)

for test_img in tqdm(test_imgs, total=len(test_imgs)):
    test_image_path = os.path.join(test_imgs_dir, test_img)
    out = model.predict(test_image_path)
    out.save(os.path.join('../inference_results/', test_img))
