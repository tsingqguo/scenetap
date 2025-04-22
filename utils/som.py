import torch

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from SoM.task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from SoM.task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from SoM.task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from SoM.task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label
import numpy as np

class SoM:
    def __init__(self, semsam_cfg, seem_cfg, semsam_ckpt, sam_ckpt, seem_ckpt):
        # Load options and models
        opt_semsam = load_opt_from_config_file(semsam_cfg)
        opt_seem = load_opt_from_config_file(seem_cfg)
        opt_seem = init_distributed_seem(opt_seem)

        self.model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
        self.model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
        self.model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                self.model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

    @torch.no_grad()
    def inference(self, image, slider, mode, alpha, label_mode, anno_mode, filter, *args, **kwargs):
        # Load the image
        _image = image
        _mask = None

        if slider < 1.5:
            model_name = 'seem'
        elif slider > 2.5:
            model_name = 'sam'
        else:
            if mode == 'Automatic':
                model_name = 'semantic-sam'
                if slider < 1.5 + 0.14:
                    level = [1]
                elif slider < 1.5 + 0.28:
                    level = [2]
                elif slider < 1.5 + 0.42:
                    level = [3]
                elif slider < 1.5 + 0.56:
                    level = [4]
                elif slider < 1.5 + 0.70:
                    level = [5]
                elif slider < 1.5 + 0.84:
                    level = [6]
                else:
                    level = [6, 1, 2, 3, 4, 5]
            else:
                model_name = 'sam'

        if label_mode == 'Alphabet':
            label_mode = 'a'
        else:
            label_mode = '1'

        text_size, hole_scale, island_scale = 640, 100, 100
        text, text_part, text_thresh = '', '', '0.0'
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            semantic = False

            if mode == "Interactive":
                labeled_array, num_features = label(np.asarray(_mask))
                spatial_masks = torch.stack([torch.from_numpy(labeled_array == i + 1) for i in range(num_features)])

            if model_name == 'semantic-sam':
                model = self.model_semsam
                output, mask = inference_semsam_m2m_auto(model, _image, level, text, text_part, text_thresh, text_size,
                                                         hole_scale, island_scale, semantic, label_mode=label_mode,
                                                         alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

            elif model_name == 'sam':
                model = self.model_sam
                if mode == "Automatic":
                    output, mask = inference_sam_m2m_auto(model, _image, text_size, label_mode, alpha, anno_mode)
                elif mode == "Interactive":
                    output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size, label_mode,
                                                                 alpha, anno_mode)

            elif model_name == 'seem':
                model = self.model_seem
                if mode == "Automatic":
                    output, mask = inference_seem_pano(model, _image, text_size, label_mode, alpha, anno_mode)
                elif mode == "Interactive":
                    output, mask = inference_seem_interactive(model, _image, spatial_masks, text_size, label_mode,
                                                              alpha, anno_mode)

            return output, mask
