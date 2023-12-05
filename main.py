import fractions
import os
from io import BytesIO

import cv2
import numpy
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms

from insightface_func.face_detect_crop_single import Face_detect_crop
from models.models import create_model
from options.test_options import TestOptions
from parsing_model.model import BiSeNet
from util.norm import SpecificNorm
from util.reverse2original import reverse2wholeimage


def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


transformer_Arcface = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


opt = TestOptions().parse()
opt.no_simswaplogo = True
opt.crop_size = 224
opt.use_mask = True
opt.name = "people"
opt.Arc_path = "arcface_model/arcface_checkpoint.tar"

print("loading model...")

model = create_model(opt)
model.eval()

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_image(file: UploadFile):
    # Open the image using Pillow
    image = Image.open(BytesIO(file.file.read()))

    # Convert the Pillow Image to a NumPy array
    image_array = numpy.array(image)

    return image_array


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post(
    "/swap-single",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def image_swap_single(source: UploadFile, target: UploadFile):
    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    mode = "None"

    spNorm = SpecificNorm()
    app = Face_detect_crop(name="antelope", root="./insightface_func/models")
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

    with torch.no_grad():
        img_a_whole = process_image(source)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(
            cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB)
        )
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()

        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        ############## Forward Pass ######################

        img_b_whole = process_image(target)

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
        # detect_results = None
        swap_result_list = []

        b_align_crop_tenor_list = []

        for b_align_crop in img_b_align_crop_list:
            b_align_crop_tenor = _totensor(
                cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB)
            )[None, ...].cuda()

            swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join("./parsing_model/checkpoint", "79999_iter.pth")
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net = None

        final_image = reverse2wholeimage(
            b_align_crop_tenor_list,
            swap_result_list,
            b_mat_list,
            crop_size,
            img_b_whole,
            None,
            os.path.join("./output", "result_whole_swapsingle.jpg"),
            True,
            pasring_model=net,
            use_mask=True,
            norm=spNorm,
        )

        response_image = Image.fromarray(final_image)

        image_bytes = BytesIO()
        response_image.save(image_bytes, format="PNG")

        image_bytes.seek(0)

        return Response(content=image_bytes.read1(), media_type="image/png")
