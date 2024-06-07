from .facexlib.utils.face_restoration_helper import FaceRestoreHelper
from .basicsr.utils import tensor2img, img2tensor
from .basicsr.archs.codeformer_arch import CodeFormer
from .basicsr.archs.gfpganv1_clean_arch import GFPGANv1Clean
from .basicsr.archs.vqvae_arch import VQVAEGANMultiHeadTransformer
from .bg_up_sampler import BackgroundUpSampler
from torchvision.transforms.functional import normalize
from typing import Literal
import torch
import os.path as osp


class FaceRestorer:
    def __init__(
        self, restorer_name: Literal['GFPGAN', 'CodeFormer', 'RestoreFormer', 'RestoreFormer++'],
        # upscale: int = 1,
        bg_up_sample: bool = False,
        device: Literal['cpu', 'gpu', 'cuda'] = 'cpu'
    ):
        self.device = device
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            use_parse=True,
            device=device,
            model_rootpath=osp.join(osp.dirname(__file__), 'models', 'FaceDetection')
        )

        if restorer_name == "GFPGAN":
            self.restorer = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True
            )

            checkpoint = torch.load(
                osp.join(osp.dirname(__file__), 'models', 'GFPGAN', 'GFPGAN_v1.4.pth')
            )['params_ema']

            self.restorer.load_state_dict(checkpoint, strict=True)
            self.restorer.eval()
            self.params = {'return_rgb': False, 'weight': 0.5}

        elif restorer_name == "CodeFormer":
            self.restorer = CodeFormer(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to('cpu')
            checkpoint = torch.load(
                osp.join(osp.dirname(__file__), 'models', 'CodeFormer', 'CodeFormer.pth')
            )['params_ema']

            self.restorer.load_state_dict(checkpoint)
            self.restorer.eval()
            self.params = {"w": 0.5, 'adain': True}

        elif restorer_name.startswith("RestoreFormer"):
            if restorer_name == "RestoreFormer":
                self.restorer = VQVAEGANMultiHeadTransformer(head_size=8, ex_multi_scale_num=0)
                checkpoint = torch.load(
                    osp.join(osp.dirname(__file__), 'models', 'RestoreFormer', 'RestoreFormer.ckpt')
                )['state_dict']
            elif restorer_name == "RestoreFormer++":
                self.restorer = VQVAEGANMultiHeadTransformer(head_size=4, ex_multi_scale_num=1)
                checkpoint = torch.load(
                    osp.join(osp.dirname(__file__), 'models', 'RestoreFormer', 'RestoreFormer++.ckpt')
                )['state_dict']
            else:
                raise NameError

            new_weights = {}
            for k, v in checkpoint.items():
                if k.startswith('vqvae.'):
                    k = k.replace('vqvae.', '')
                new_weights[k] = v
            self.restorer.load_state_dict(new_weights, strict=False)
            self.restorer.eval()
            self.params = {}

        else:
            raise NameError

        if bg_up_sample:
            self.bg_up_sampler = BackgroundUpSampler(device=device)
        else:
            self.bg_up_sampler = None

    def restore(self, img, out_scale, only_center_face=False):
        self.face_helper.clean_all()
        self.face_helper.set_upscale_factor(out_scale)

        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.restorer(cropped_face_t, **self.params)[0]
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(error)
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if self.bg_up_sampler is not None:
            bg_img = self.bg_up_sampler.enhance(img, out_scale=out_scale)
        else:
            bg_img = None

        self.face_helper.get_inverse_affine(None)
        restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)

        return restored_img
