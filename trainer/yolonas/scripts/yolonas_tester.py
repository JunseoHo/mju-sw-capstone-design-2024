from super_gradients.training import models
from tqdm.auto import tqdm
import torch
import os


model_name = 'yolo_nas_s'
pth_path = '../checkpoints/yolo_nas_s/RUN_20240508_175406_865478/ckpt_best.pth'
classes = ['head', 'helmet']

model = models.get(
    model_name=model_name,
    checkpoint_path=pth_path,
    num_classes=len(classes)
).to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

test_imgs_dir = '../dataset/test/images/'
test_images = os.listdir(test_imgs_dir)

for test_image in tqdm(test_images, total=len(test_images)):
    test_image_path = os.path.join(test_imgs_dir, test_image)
    out = model.predict(test_image_path)
    print(out.image)
    print(out.prediction.bboxes_xyxy)
    out.save(os.path.join('../inference_results/', test_image))
