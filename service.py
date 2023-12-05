import fractions
import os

import bentoml
import cv2
import numpy
import torch
import torch.nn.functional as F
from bentoml.io import Image, Multipart
from PIL import Image
from torchvision import transforms

from insightface_func.face_detect_crop_single import Face_detect_crop
from util.norm import SpecificNorm
from util.reverse2original import reverse2wholeimage

simswap_runner = bentoml.models.get("simswap:latest").to_runner()
bisenet_runner = bentoml.models.get("bisenet:latest").to_runner()

svc = bentoml.Service(name="simswap", runners=[simswap_runner, bisenet_runner])


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


@svc.api(
    input=Multipart(source=bentoml.io.Image(), target=bentoml.io.Image()),
    output=bentoml.io.Image(),
)
def image_swap_single(source, target):
    crop_size = 224

    torch.nn.Module.dump_patches = True
    mode = "None"

    spNorm = SpecificNorm()
    app = Face_detect_crop(name="antelope", root="./insightface_func/models")
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

    with torch.no_grad():
        img_a_whole = numpy.array(source)
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
        latend_id = simswap_runner.netArc.run(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        ############## Forward Pass ######################

        img_b_whole = numpy.array(target)

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
        # detect_results = None
        swap_result_list = []

        b_align_crop_tenor_list = []

        for b_align_crop in img_b_align_crop_list:
            b_align_crop_tenor = _totensor(
                cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB)
            )[None, ...].cuda()

            swap_result = simswap_runner.run(
                None, b_align_crop_tenor, latend_id, None, True
            )[0]
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        final_image = reverse2wholeimage(
            b_align_crop_tenor_list,
            swap_result_list,
            b_mat_list,
            crop_size,
            img_b_whole,
            None,
            os.path.join("./output", "result_whole_swapsingle.jpg"),
            True,
            pasring_model=bisenet_runner.run,
            use_mask=True,
            norm=spNorm,
        )

        return final_image
        print(" ")

        print("************ Done ! ************")
