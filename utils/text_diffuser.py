import re

import torch
import string
import copy

import numpy as np
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def get_min_bounding_rectangle(small_region):
    """Calculate the minimum bounding rectangle for the given set of points."""
    small_region = np.array(small_region)

    # Get the bounding box of the region
    min_x, min_y = np.min(small_region, axis=0)
    max_x, max_y = np.max(small_region, axis=0)

    return min_x, min_y, max_x, max_y


def get_scaled_square_region(min_x, min_y, max_x, max_y, img_width, img_height, scale_factor=2.0):
    """Calculate a square region that contains the bounding rectangle and is scaled."""
    # Calculate the size of the bounding box
    width = max_x - min_x
    height = max_y - min_y
    # Find the larger dimension to make it a square
    side_length = max(width, height) * scale_factor

    # Calculate the center of the bounding box
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    # Calculate the new square region coordinates
    half_side = side_length // 2
    left = max(center_x - half_side, 0)  # Ensure we don't go outside the left image boundary
    right = min(center_x + half_side, img_width)  # Ensure we don't go outside the right image boundary
    top = max(center_y - half_side, 0)  # Ensure we don't go outside the top image boundary
    bottom = min(center_y + half_side, img_height)  # Ensure we don't go outside the bottom image boundary

    # make it a square, pay attention to edge cases
    if right - left > bottom - top:
        diff = right - left - (bottom - top)
        if top - diff / 2 < 0:
            top = 0
            bottom = right - left
        elif bottom + diff / 2 > img_height:
            bottom = img_height
            top = bottom - right + left
        else:
            top -= diff / 2
            bottom += diff / 2
    elif right - left < bottom - top:
        diff = bottom - top - (right - left)
        if left - diff / 2 < 0:
            left = 0
            right = bottom - top
        elif right + diff / 2 > img_width:
            right = img_width
            left = right - bottom + top
        else:
            left -= diff / 2
            right += diff / 2

    print('right - left', int(right) - int(left))
    print('bottom - top', int(bottom) - int(top))

    # Return the new square region's coordinates as the bounding box (left, top, right, bottom)
    return int(left), int(top), int(right), int(bottom)


class TextDiffuser(object):
    def __init__(self):
        super(TextDiffuser, self).__init__()
        alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
        # alphabet contains
        '''
        0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
        '''

        #### import diffusion models
        self.text_encoder = CLIPTextModel.from_pretrained(
            'JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="text_encoder"
        ).cuda().half()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'runwayml/stable-diffusion-v1-5', subfolder="tokenizer"
        )

        #### additional tokens are introduced, including coordinate tokens and character tokens
        print('***************')
        print(len(self.tokenizer))
        for i in range(520):
            self.tokenizer.add_tokens(['l' + str(i)])  # left
            self.tokenizer.add_tokens(['t' + str(i)])  # top
            self.tokenizer.add_tokens(['r' + str(i)])  # width
            self.tokenizer.add_tokens(['b' + str(i)])  # height
        for c in alphabet:
            self.tokenizer.add_tokens([f'[{c}]'])
        print(len(self.tokenizer))
        print('***************')

        self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae").half().cuda()
        self.unet = UNet2DConditionModel.from_pretrained(
            'JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="unet"
        ).half().cuda()
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        self.global_dict = {}
        #### for interactive
        # stack = []
        # state = 0
        self.font = ImageFont.truetype("fonts/arial.ttf", 20)

    def get_layout_image(self, ocrs):

        font_layout = ImageFont.truetype('fonts/arial.ttf', 16)

        blank = Image.new('RGB', (256, 256), (0, 0, 0))
        draw = ImageDraw.ImageDraw(blank)

        for line in ocrs.split('\n'):
            line = line.strip()

            if len(line) == 0:
                break

            pred = ' '.join(line.split()[:-1])
            box = line.split()[-1]
            l, t, r, b = [int(i) * 2 for i in box.split(',')]  # the size of canvas is 256x256
            draw.rectangle([(l, t), (r, b)], outline="red")
            draw.text((l, t), pred, font=font_layout)

        return blank

    def to_tensor(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise TypeError("Error")

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image)

        return tensor

    def get_pixels(self, i, orig_i, radio, t, guest_id, text_position):

        print('hi1 ', i)
        print('hi2 ', orig_i)

        width, height = Image.open(i).convert("RGB").size

        # register
        if guest_id == '-1':  # register for the first time
            seed = str(int(time.time()))
            self.global_dict[str(seed)] = {
                'state': 0,
                'stack': [],
                'image_id': [list(Image.open(i).convert("RGB").resize((512, 512)).getdata())]
                # an image has been recorded
            }
            guest_id = str(seed)
        else:
            seed = guest_id

        if type(i) == str:
            i = Image.open(i).convert("RGB")
            i = i.resize((512, 512))

        images = self.global_dict[str(seed)]['image_id']
        flag = False
        for image in images:
            if image == list(i.getdata()):
                print('find it')
                flag = True
                break

        if not flag:
            self.global_dict[str(seed)]['image_id'] = [list(i.getdata())]
            self.global_dict[str(seed)]['stack'] = []
            self.global_dict[str(seed)]['state'] = 0
            orig_i = i
        else:

            if orig_i is not None:
                orig_i = Image.open(orig_i).convert("RGB")
                orig_i = orig_i.resize((512, 512))
            else:
                orig_i = i
                self.global_dict[guest_id]['stack'] = []
                self.global_dict[guest_id]['state'] = 0

        print('hello ', text_position)

        if radio == 'Two Points':

            if self.global_dict[guest_id]['state'] == 0:
                self.global_dict[guest_id]['stack'].append(
                    (text_position, t)
                )
                print(text_position, self.global_dict[guest_id]['stack'])
                self.global_dict[guest_id]['state'] = 1
            else:

                (_, t) = self.global_dict[guest_id]['stack'].pop()
                x, y = _
                self.global_dict[guest_id]['stack'].append(
                    ((x, y, text_position[0], text_position[1]), t)
                )
                self.global_dict[guest_id]['state'] = 0

            image = copy.deepcopy(orig_i)
            draw = ImageDraw.Draw(image)

            for items in self.global_dict[guest_id]['stack']:
                text_position, t = items
                if len(text_position) == 2:
                    x, y = text_position

                    x = int(512 * x / width)
                    y = int(512 * y / height)

                    text_color = (255, 0, 0)
                    draw.text((x + 2, y), t, font=self.font, fill=text_color)
                    r = 4
                    leftUpPoint = (x - r, y - r)
                    rightDownPoint = (x + r, y + r)
                    draw.ellipse((leftUpPoint, rightDownPoint), fill='red')
                elif len(text_position) == 4:
                    x0, y0, x1, y1 = text_position

                    x0 = int(512 * x0 / width)
                    x1 = int(512 * x1 / width)
                    y0 = int(512 * y0 / height)
                    y1 = int(512 * y1 / height)

                    text_color = (255, 0, 0)
                    draw.text((x0 + 2, y0), t, font=self.font, fill=text_color)
                    r = 4
                    leftUpPoint = (x0 - r, y0 - r)
                    rightDownPoint = (x0 + r, y0 + r)
                    draw.ellipse((leftUpPoint, rightDownPoint), fill='red')
                    draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0))

        elif radio == 'Four Points':

            if self.global_dict[guest_id]['state'] == 0:
                self.global_dict[guest_id]['stack'].append(
                    (text_position, t)
                )
                print(text_position, self.global_dict[guest_id]['stack'])
                self.global_dict[guest_id]['state'] = 1
            elif self.global_dict[guest_id]['state'] == 1:
                (_, t) = self.global_dict[guest_id]['stack'].pop()
                x, y = _
                self.global_dict[guest_id]['stack'].append(
                    ((x, y, text_position[0], text_position[1]), t)
                )
                self.global_dict[guest_id]['state'] = 2
            elif self.global_dict[guest_id]['state'] == 2:
                (_, t) = self.global_dict[guest_id]['stack'].pop()
                x0, y0, x1, y1 = _
                self.global_dict[guest_id]['stack'].append(
                    ((x0, y0, x1, y1, text_position[0], text_position[1]), t)
                )
                self.global_dict[guest_id]['state'] = 3
            elif self.global_dict[guest_id]['state'] == 3:
                (_, t) = self.global_dict[guest_id]['stack'].pop()
                x0, y0, x1, y1, x2, y2 = _
                self.global_dict[guest_id]['stack'].append(
                    ((x0, y0, x1, y1, x2, y2, text_position[0], text_position[1]), t)
                )
                self.global_dict[guest_id]['state'] = 0

            image = copy.deepcopy(orig_i)
            draw = ImageDraw.Draw(image)

            for items in self.global_dict[guest_id]['stack']:
                text_position, t = items
                if len(text_position) == 2:
                    x, y = text_position

                    x = int(512 * x / width)
                    y = int(512 * y / height)

                    text_color = (255, 0, 0)
                    draw.text((x + 2, y), t, font=self.font, fill=text_color)
                    r = 4
                    leftUpPoint = (x - r, y - r)
                    rightDownPoint = (x + r, y + r)
                    draw.ellipse((leftUpPoint, rightDownPoint), fill='red')
                elif len(text_position) == 4:
                    x0, y0, x1, y1 = text_position
                    text_color = (255, 0, 0)
                    draw.text((x0 + 2, y0), t, font=self.font, fill=text_color)
                    r = 4
                    leftUpPoint = (x0 - r, y0 - r)
                    rightDownPoint = (x0 + r, y0 + r)
                    draw.ellipse((leftUpPoint, rightDownPoint), fill='red')
                    draw.line(((x0, y0), (x1, y1)), fill=(255, 0, 0))
                elif len(text_position) == 6:
                    x0, y0, x1, y1, x2, y2 = text_position
                    text_color = (255, 0, 0)
                    draw.text((x0 + 2, y0), t, font=self.font, fill=text_color)
                    r = 4
                    leftUpPoint = (x0 - r, y0 - r)
                    rightDownPoint = (x0 + r, y0 + r)
                    draw.ellipse((leftUpPoint, rightDownPoint), fill='red')
                    draw.line(((x0, y0), (x1, y1)), fill=(255, 0, 0))
                    draw.line(((x1, y1), (x2, y2)), fill=(255, 0, 0))
                elif len(text_position) == 8:
                    x0, y0, x1, y1, x2, y2, x3, y3 = text_position
                    text_color = (255, 0, 0)
                    draw.text((x0 + 2, y0), t, font=self.font, fill=text_color)
                    r = 4
                    leftUpPoint = (x0 - r, y0 - r)
                    rightDownPoint = (x0 + r, y0 + r)
                    draw.ellipse((leftUpPoint, rightDownPoint), fill='red')
                    draw.line(((x0, y0), (x1, y1)), fill=(255, 0, 0))
                    draw.line(((x1, y1), (x2, y2)), fill=(255, 0, 0))
                    draw.line(((x2, y2), (x3, y3)), fill=(255, 0, 0))
                    draw.line(((x3, y3), (x0, y0)), fill=(255, 0, 0))

        print('stack', self.global_dict[guest_id]['stack'])

        self.global_dict[str(seed)]['image_id'].append(list(image.getdata()))

        # image.save('image_draw.png')
        return image, orig_i, seed

    def text_to_image_regional(
            self,
            guest_id,
            i,
            orig_i,
            prompt,
            keywords,
            positive_prompt,
            radio,  #
            slider_step,
            slider_guidance,
            slider_batch,
            slider_temperature,
            slider_natural,
            scale_factor=2.0,
            answer_instruct=False
    ):

        # print(type(i))
        # exit(0)

        print(
            f'[info] Prompt: {prompt} | Keywords: {keywords} | Radio: {radio} | Steps: {slider_step} | Guidance: {slider_guidance} | Natural: {slider_natural}')

        # global stack
        # global state

        if len(positive_prompt.strip()) != 0:
            prompt += positive_prompt

        instruct_flag = False
        with torch.no_grad():
            time1 = time.time()
            user_prompt = prompt

            if slider_natural:
                user_prompt = f'{user_prompt}'
                composed_prompt = user_prompt
                prompt = self.tokenizer.encode(user_prompt)
                layout_image = None
            else:
                if guest_id not in self.global_dict or len(self.global_dict[guest_id]['stack']) == 0:

                    if len(keywords.strip()) == 0:
                        template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. All keywords are included in the caption. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {user_prompt}'
                    else:
                        keywords = keywords.split('/')
                        keywords = [i.strip() for i in keywords]
                        template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. In addition, we also provide all keywords at random order for reference. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {prompt}. Keywords: {str(keywords)}'

                    msg = template
                    conv = get_conversation_template(m1_model_path)
                    conv.append_message(conv.roles[0], msg)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    inputs = m1_tokenizer([prompt], return_token_type_ids=False)
                    inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
                    output_ids = m1_model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=slider_temperature,
                        repetition_penalty=1.0,
                        max_new_tokens=512,
                    )

                    if m1_model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
                    outputs = m1_tokenizer.decode(
                        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                    )
                    print(f"[{conv.roles[0]}]\n{msg}")
                    print(f"[{conv.roles[1]}]\n{outputs}")
                    layout_image = get_layout_image(outputs)

                    ocrs = outputs.split('\n')
                    time2 = time.time()
                    print(time2 - time1)

                    # user_prompt = prompt
                    current_ocr = ocrs

                    ocr_ids = []
                    print('user_prompt', user_prompt)
                    print('current_ocr', current_ocr)

                    for ocr in current_ocr:
                        ocr = ocr.strip()

                        if len(ocr) == 0 or '###' in ocr or '.com' in ocr:
                            continue

                        items = ocr.split()
                        pred = ' '.join(items[:-1])
                        box = items[-1]

                        l, t, r, b = box.split(',')
                        l, t, r, b = int(l), int(t), int(r), int(b)
                        ocr_ids.extend(['l' + str(l), 't' + str(t), 'r' + str(r), 'b' + str(b)])

                        char_list = list(pred)
                        char_list = [f'[{i}]' for i in char_list]
                        ocr_ids.extend(char_list)
                        ocr_ids.append(self.tokenizer.eos_token_id)

                    caption_ids = self.tokenizer(
                        user_prompt, truncation=True, return_tensors="pt"
                    ).input_ids[0].tolist()

                    try:
                        ocr_ids = self.tokenizer.encode(ocr_ids)
                        prompt = caption_ids + ocr_ids
                    except:
                        prompt = caption_ids

                    user_prompt = self.tokenizer.decode(prompt)
                    composed_prompt = self.tokenizer.decode(prompt)

                else:
                    user_prompt += ' <|endoftext|><|startoftext|>'
                    layout_image = None

                    image_mask = Image.new('L', (512, 512), 0)
                    draw = ImageDraw.Draw(image_mask)

                    for items in self.global_dict[guest_id]['stack']:
                        position, text = items

                        # feature_mask
                        # masked_feature

                        if len(position) == 2:
                            x, y = position
                            x = x // 4
                            y = y // 4
                            text_str = ' '.join([f'[{c}]' for c in list(text)])
                            user_prompt += f' l{x} t{y} {text_str} <|endoftext|>'

                        elif len(position) == 4:

                            x0, y0, x1, y1 = position

                            # region diffusion
                            image_origin = Image.open(orig_i).convert("RGB")
                            img_width, img_height = image_origin.size
                            if answer_instruct and (x1 - x0) > img_width // 4 and (y1 - y0) > img_height // 4:
                                instruct_flag = True

                            small_region_coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                            # Get the minimum bounding rectangle of the small region
                            min_x, min_y, max_x, max_y = get_min_bounding_rectangle(small_region_coords)
                            # Get the scaled square region that fully contains the bounding rectangle
                            larger_region_bbox = get_scaled_square_region(min_x, min_y, max_x, max_y, img_width,
                                                                          img_height, scale_factor)
                            larger_region_bbox = tuple([int(i) for i in larger_region_bbox])
                            assert larger_region_bbox[2] - larger_region_bbox[0] == larger_region_bbox[3] - \
                                   larger_region_bbox[1]
                            edge_length = larger_region_bbox[2] - larger_region_bbox[0]
                            print("edge_length", edge_length)
                            # get coordinates of the smaller region from original image coordinate to larger region coordinate
                            x0, y0, x1, y1 = (x0 - larger_region_bbox[0]) / edge_length * img_width, \
                                             (y0 - larger_region_bbox[1]) / edge_length * img_height, \
                                             (x1 - larger_region_bbox[0]) / edge_length * img_width, \
                                             (y1 - larger_region_bbox[1]) / edge_length * img_height
                            x0, y0, x1, y1 = (int(x0 / img_width * 512), int(y0 / img_height * 512),
                                              int(x1 / img_width * 512), int(y1 / img_height * 512))
                            # region diffusion end 1
                            x0 = x0 // 4
                            y0 = y0 // 4
                            x1 = x1 // 4
                            y1 = y1 // 4

                            if instruct_flag:
                                text = "answer:" + text
                                user_prompt = re.sub(r'''(["'])(.*?)\1''', r'\1answer:\2\1', user_prompt)
                            text_str = ' '.join([f'[{c}]' for c in list(text)])
                            user_prompt += f' l{x0} t{y0} r{x1} b{y1} {text_str} <|endoftext|>'

                            draw.rectangle((x0 * 4, y0 * 4, x1 * 4, y1 * 4), fill=1)
                            print('prompt ', user_prompt)

                        elif len(position) == 8:  # four points
                            x0, y0, x1, y1, x2, y2, x3, y3 = position

                            # region diffusion
                            img_width, img_height = image_mask.size
                            small_region_coords = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
                            # Get the minimum bounding rectangle of the small region
                            min_x, min_y, max_x, max_y = get_min_bounding_rectangle(small_region_coords)
                            # Get the scaled square region that fully contains the bounding rectangle
                            larger_region_bbox = get_scaled_square_region(min_x, min_y, max_x, max_y, img_width,
                                                                          img_height, scale_factor)
                            larger_region_bbox = tuple([int(i) for i in larger_region_bbox])
                            assert larger_region_bbox[2] - larger_region_bbox[0] == larger_region_bbox[3] - \
                                   larger_region_bbox[1]
                            edge_length = larger_region_bbox[2] - larger_region_bbox[0]
                            print("edge_length", edge_length)
                            # get coordinates of the smaller region from original image coordinate to larger region coordinate
                            x0, y0, x1, y1, x2, y2, x3, y3 = (x0 - larger_region_bbox[0]) / edge_length * img_width, \
                                                             (y0 - larger_region_bbox[1]) / edge_length * img_height, \
                                                             (x1 - larger_region_bbox[0]) / edge_length * img_width, \
                                                             (y1 - larger_region_bbox[1]) / edge_length * img_height, \
                                                             (x2 - larger_region_bbox[0]) / edge_length * img_width, \
                                                             (y2 - larger_region_bbox[1]) / edge_length * img_height, \
                                                             (x3 - larger_region_bbox[0]) / edge_length * img_width, \
                                                             (y3 - larger_region_bbox[1]) / edge_length * img_height

                            # region diffusion end 1

                            draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=1)
                            x0 = x0 // 4
                            y0 = y0 // 4
                            x1 = x1 // 4
                            y1 = y1 // 4
                            x2 = x2 // 4
                            y2 = y2 // 4
                            x3 = x3 // 4
                            y3 = y3 // 4
                            xmin = min(x0, x1, x2, x3)
                            ymin = min(y0, y1, y2, y3)
                            xmax = max(x0, x1, x2, x3)
                            ymax = max(y0, y1, y2, y3)
                            text_str = ' '.join([f'[{c}]' for c in list(text)])
                            user_prompt += f' l{xmin} t{ymin} r{xmax} b{ymax} {text_str} <|endoftext|>'

                            print('prompt ', user_prompt)

                        prompt = self.tokenizer.encode(user_prompt)
                        composed_prompt = self.tokenizer.decode(prompt)

            prompt = prompt[:77]
            while len(prompt) < 77:
                prompt.append(self.tokenizer.pad_token_id)

            prompts_cond = prompt
            prompts_nocond = [self.tokenizer.pad_token_id] * 77

            prompts_cond = [prompts_cond] * slider_batch
            prompts_nocond = [prompts_nocond] * slider_batch

            prompts_cond = torch.Tensor(prompts_cond).long().cuda()
            prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()

            scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
            scheduler.set_timesteps(slider_step)
            noise = torch.randn((slider_batch, 4, 64, 64)).to("cuda").half()
            input = noise

            encoder_hidden_states_cond = self.text_encoder(prompts_cond)[0].half()
            encoder_hidden_states_nocond = self.text_encoder(prompts_nocond)[0].half()

            image = Image.open(orig_i).convert("RGB").resize((512, 512))

            # image, mask ready
            # region diffusion
            image = Image.open(orig_i).convert("RGB")
            # Crop the larger region from the original image
            larger_region = image.crop(larger_region_bbox)
            larger_region_shape = larger_region.size
            # # reshape to 512x512
            larger_region = larger_region.resize((512, 512))

            image = larger_region
            # image_mask = larger_region_mask

            # image.save('larger_region.png')
            # image_mask.save('larger_region_mask.png')

            # region diffusion end 2

            image_mask = torch.Tensor(np.array(image_mask)).float().half().cuda()
            image_mask = image_mask.unsqueeze(0).unsqueeze(0).repeat(slider_batch, 1, 1, 1)

            image_tensor = self.to_tensor(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
            print(f'image_tensor.shape {image_tensor.shape}')
            masked_image = image_tensor * (1 - image_mask)
            masked_feature = self.vae.encode(masked_image.half()).latent_dist.sample()
            masked_feature = masked_feature * self.vae.config.scaling_factor
            masked_feature = masked_feature.half()
            print(f'masked_feature.shape {masked_feature.shape}')

            feature_mask = torch.nn.functional.interpolate(image_mask, size=(64, 64), mode='nearest').cuda()

            for t in tqdm(scheduler.timesteps):
                with torch.no_grad():  # classifier free guidance

                    noise_pred_cond = self.unet(sample=input, timestep=t,
                                                encoder_hidden_states=encoder_hidden_states_cond[:slider_batch],
                                                feature_mask=feature_mask,
                                                masked_feature=masked_feature).sample  # b, 4, 64, 64
                    noise_pred_uncond = self.unet(sample=input, timestep=t,
                                                  encoder_hidden_states=encoder_hidden_states_nocond[:slider_batch],
                                                  feature_mask=feature_mask,
                                                  masked_feature=masked_feature).sample  # b, 4, 64, 64
                    noisy_residual = noise_pred_uncond + slider_guidance * (
                            noise_pred_cond - noise_pred_uncond)  # b, 4, 64, 64
                    input = scheduler.step(noisy_residual, t, input).prev_sample
                    del noise_pred_cond
                    del noise_pred_uncond

                    torch.cuda.empty_cache()

            # decode
            input = 1 / self.vae.config.scaling_factor * input
            images = self.vae.decode(input, return_dict=False)[0]
            width, height = 512, 512
            results = []
            new_image = Image.new('RGB', (2 * width, 2 * height))
            for index, image in enumerate(images.cpu().float()):
                image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                results.append(image)
                row = index // 2
                col = index % 2
                new_image.paste(image, (col * width, row * height))
            # os.system('nvidia-smi')
            torch.cuda.empty_cache()
            # os.system('nvidia-smi')

            # region diffusion
            # reshape back
            image_origin = Image.open(orig_i).convert("RGB")
            for i in range(len(results)):
                image_origin = Image.open(orig_i).convert("RGB")
                results[i] = results[i].resize(larger_region_shape)
                image_origin.paste(results[i], (larger_region_bbox[0], larger_region_bbox[1]))
                results[i] = image_origin
            # region diffusion end 3

            return tuple(results), composed_prompt

    def text_to_image(
            self,
            guest_id,
            i,
            orig_i,
            prompt,
            keywords,
            positive_prompt,
            radio,  #
            slider_step,
            slider_guidance,
            slider_batch,
            slider_temperature,
            slider_natural,
    ):

        # print(type(i))
        # exit(0)

        print(
            f'[info] Prompt: {prompt} | Keywords: {keywords} | Radio: {radio} | Steps: {slider_step} | Guidance: {slider_guidance} | Natural: {slider_natural}')

        # global stack
        # global state

        if len(positive_prompt.strip()) != 0:
            prompt += positive_prompt

        with torch.no_grad():
            time1 = time.time()
            user_prompt = prompt

            if slider_natural:
                user_prompt = f'{user_prompt}'
                composed_prompt = user_prompt
                prompt = self.tokenizer.encode(user_prompt)
                layout_image = None
            else:
                if guest_id not in self.global_dict or len(self.global_dict[guest_id]['stack']) == 0:

                    if len(keywords.strip()) == 0:
                        template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. All keywords are included in the caption. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {user_prompt}'
                    else:
                        keywords = keywords.split('/')
                        keywords = [i.strip() for i in keywords]
                        template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. In addition, we also provide all keywords at random order for reference. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {prompt}. Keywords: {str(keywords)}'

                    msg = template
                    conv = get_conversation_template(m1_model_path)
                    conv.append_message(conv.roles[0], msg)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    inputs = m1_tokenizer([prompt], return_token_type_ids=False)
                    inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
                    output_ids = m1_model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=slider_temperature,
                        repetition_penalty=1.0,
                        max_new_tokens=512,
                    )

                    if m1_model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
                    outputs = m1_tokenizer.decode(
                        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                    )
                    print(f"[{conv.roles[0]}]\n{msg}")
                    print(f"[{conv.roles[1]}]\n{outputs}")
                    layout_image = get_layout_image(outputs)

                    ocrs = outputs.split('\n')
                    time2 = time.time()
                    print(time2 - time1)

                    # user_prompt = prompt
                    current_ocr = ocrs

                    ocr_ids = []
                    print('user_prompt', user_prompt)
                    print('current_ocr', current_ocr)

                    for ocr in current_ocr:
                        ocr = ocr.strip()

                        if len(ocr) == 0 or '###' in ocr or '.com' in ocr:
                            continue

                        items = ocr.split()
                        pred = ' '.join(items[:-1])
                        box = items[-1]

                        l, t, r, b = box.split(',')
                        l, t, r, b = int(l), int(t), int(r), int(b)
                        ocr_ids.extend(['l' + str(l), 't' + str(t), 'r' + str(r), 'b' + str(b)])

                        char_list = list(pred)
                        char_list = [f'[{i}]' for i in char_list]
                        ocr_ids.extend(char_list)
                        ocr_ids.append(self.tokenizer.eos_token_id)

                    caption_ids = self.tokenizer(
                        user_prompt, truncation=True, return_tensors="pt"
                    ).input_ids[0].tolist()

                    try:
                        ocr_ids = self.tokenizer.encode(ocr_ids)
                        prompt = caption_ids + ocr_ids
                    except:
                        prompt = caption_ids

                    user_prompt = self.tokenizer.decode(prompt)
                    composed_prompt = self.tokenizer.decode(prompt)

                else:
                    user_prompt += ' <|endoftext|><|startoftext|>'
                    layout_image = None

                    image_mask = Image.new('L', (512, 512), 0)
                    draw = ImageDraw.Draw(image_mask)

                    for items in self.global_dict[guest_id]['stack']:
                        position, text = items

                        # feature_mask
                        # masked_feature

                        if len(position) == 2:
                            x, y = position
                            x = x // 4
                            y = y // 4
                            text_str = ' '.join([f'[{c}]' for c in list(text)])
                            user_prompt += f' l{x} t{y} {text_str} <|endoftext|>'

                        elif len(position) == 4:
                            x0, y0, x1, y1 = position
                            x0 = x0 // 4
                            y0 = y0 // 4
                            x1 = x1 // 4
                            y1 = y1 // 4
                            text_str = ' '.join([f'[{c}]' for c in list(text)])
                            user_prompt += f' l{x0} t{y0} r{x1} b{y1} {text_str} <|endoftext|>'

                            draw.rectangle((x0 * 4, y0 * 4, x1 * 4, y1 * 4), fill=1)
                            print('prompt ', user_prompt)

                        elif len(position) == 8:  # four points
                            x0, y0, x1, y1, x2, y2, x3, y3 = position
                            draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=1)
                            x0 = x0 // 4
                            y0 = y0 // 4
                            x1 = x1 // 4
                            y1 = y1 // 4
                            x2 = x2 // 4
                            y2 = y2 // 4
                            x3 = x3 // 4
                            y3 = y3 // 4
                            xmin = min(x0, x1, x2, x3)
                            ymin = min(y0, y1, y2, y3)
                            xmax = max(x0, x1, x2, x3)
                            ymax = max(y0, y1, y2, y3)
                            text_str = ' '.join([f'[{c}]' for c in list(text)])
                            user_prompt += f' l{xmin} t{ymin} r{xmax} b{ymax} {text_str} <|endoftext|>'

                            print('prompt ', user_prompt)

                        prompt = self.tokenizer.encode(user_prompt)
                        composed_prompt = self.tokenizer.decode(prompt)

            prompt = prompt[:77]
            while len(prompt) < 77:
                prompt.append(self.tokenizer.pad_token_id)

            prompts_cond = prompt
            prompts_nocond = [self.tokenizer.pad_token_id] * 77

            prompts_cond = [prompts_cond] * slider_batch
            prompts_nocond = [prompts_nocond] * slider_batch

            prompts_cond = torch.Tensor(prompts_cond).long().cuda()
            prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()

            scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
            scheduler.set_timesteps(slider_step)
            noise = torch.randn((slider_batch, 4, 64, 64)).to("cuda").half()
            input = noise

            encoder_hidden_states_cond = self.text_encoder(prompts_cond)[0].half()
            encoder_hidden_states_nocond = self.text_encoder(prompts_nocond)[0].half()

            image_mask = torch.Tensor(np.array(image_mask)).float().half().cuda()
            image_mask = image_mask.unsqueeze(0).unsqueeze(0).repeat(slider_batch, 1, 1, 1)

            image = Image.open(orig_i).convert("RGB").resize((512, 512))
            image_tensor = self.to_tensor(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
            print(f'image_tensor.shape {image_tensor.shape}')
            masked_image = image_tensor * (1 - image_mask)
            masked_feature = self.vae.encode(masked_image.half()).latent_dist.sample()
            masked_feature = masked_feature * self.vae.config.scaling_factor
            masked_feature = masked_feature.half()
            print(f'masked_feature.shape {masked_feature.shape}')

            feature_mask = torch.nn.functional.interpolate(image_mask, size=(64, 64), mode='nearest').cuda()

            for t in tqdm(scheduler.timesteps):
                with torch.no_grad():  # classifier free guidance

                    noise_pred_cond = self.unet(sample=input, timestep=t,
                                                encoder_hidden_states=encoder_hidden_states_cond[:slider_batch],
                                                feature_mask=feature_mask,
                                                masked_feature=masked_feature).sample  # b, 4, 64, 64
                    noise_pred_uncond = self.unet(sample=input, timestep=t,
                                                  encoder_hidden_states=encoder_hidden_states_nocond[:slider_batch],
                                                  feature_mask=feature_mask,
                                                  masked_feature=masked_feature).sample  # b, 4, 64, 64
                    noisy_residual = noise_pred_uncond + slider_guidance * (
                            noise_pred_cond - noise_pred_uncond)  # b, 4, 64, 64
                    input = scheduler.step(noisy_residual, t, input).prev_sample
                    del noise_pred_cond
                    del noise_pred_uncond

                    torch.cuda.empty_cache()

            # decode
            input = 1 / self.vae.config.scaling_factor * input
            images = self.vae.decode(input, return_dict=False)[0]
            width, height = 512, 512
            results = []
            new_image = Image.new('RGB', (2 * width, 2 * height))
            for index, image in enumerate(images.cpu().float()):
                image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                results.append(image)
                row = index // 2
                col = index % 2
                new_image.paste(image, (col * width, row * height))
            # os.system('nvidia-smi')
            torch.cuda.empty_cache()
            # os.system('nvidia-smi')
            return tuple(results), composed_prompt

    def generate(
            self,
            key_points,
            image: str,
            text: str,
            prompt: str,
            radio: str,
            keywords: str = "",
            positive_prompt: str = "",
            guest_id: str = "-1",
            scale_factor=2.,
            regional_diffusion=True,
            answer_instruct=False,
    ) -> Image.Image:
        """
        Generate an image with text diffused according to the input positions.

        Args:
            positions (list): A list of positions [(x1, y1), (x2, y2)] or [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
            image (PIL.Image.Image): Input image on which text is placed.
            text (str): Text to be diffused onto the image.
            prompt (str): Prompt that may influence how the text is styled or placed.

        Returns:
            PIL.Image.Image: The output image with the text diffused.
        """

        for kp in key_points:
            i, orig_i, guest_id = self.get_pixels(
                i=image,
                orig_i=image,
                radio=radio,
                guest_id=guest_id,
                t=text,
                text_position=kp,
            )

        # Generate the image with the text diffused
        if regional_diffusion:
            generate_image = self.text_to_image_regional(
                guest_id=guest_id,
                i=image,
                orig_i=image,
                prompt=prompt,
                keywords=keywords,
                positive_prompt=positive_prompt,
                radio=radio,
                slider_step=20,
                slider_guidance=2.5,
                slider_batch=5,
                slider_temperature=1,
                slider_natural=False,
                scale_factor=scale_factor,
                answer_instruct=answer_instruct
            )
        else:
            generate_image = self.text_to_image(
                guest_id=guest_id,
                i=image,
                orig_i=image,
                prompt=prompt,
                keywords=keywords,
                positive_prompt=positive_prompt,
                radio=radio,
                slider_step=20,
                slider_guidance=2.5,
                slider_batch=1,
                slider_temperature=1,
                slider_natural=False
            )

        return generate_image
