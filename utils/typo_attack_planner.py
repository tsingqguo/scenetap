import base64
import os
import time
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Function to encode the image
from pydantic import BaseModel

from utils.get_rectangle_by_mask import largest_inscribed_rectangle
# from utils.som import SoM
from utils.completion_request import CompletionRequest


from utils.text_diffuser import TextDiffuser


class PlanSom(BaseModel):
    image_analysis: str
    correct_answer: str
    incorrect_answer: str
    adversarial_text: str
    text_position_number: int
    text_placement: str
    short_caption_with_adversarial_text: str


class PlanSomAdjust(BaseModel):
    adjust_explanation: str
    adjust_plan: PlanSom


def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def format_instance_json(instance):
    # Get the attribute names from the class definition
    attributes = instance.__class__.__annotations__.keys()

    # Retrieve the values from the instance
    values = {attr: getattr(instance, attr) for attr in attributes}

    return values


def find_text_region(text, left, top, right, bottom, font_path='./fonts/arialbd.ttf', font_size=20, aspect_ratio_threshold=0.1):
    # Load the font (you may need to provide the correct font path)
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the width and height of the original region
    w = right - left
    h = bottom - top

    # Get the text size (width and height)
    text_width, text_height = font.getsize(text)

    # Calculate text aspect ratio
    text_aspect_ratio = text_height / text_width

    # Calculate the region aspect ratio
    region_aspect_ratio = h / w

    # Compare the two aspect ratios
    aspect_ratio_difference = abs(region_aspect_ratio - text_aspect_ratio)

    if aspect_ratio_difference > aspect_ratio_threshold:
        # If the aspect ratios differ too much, adjust the region
        if text_aspect_ratio > region_aspect_ratio:
            # Text is taller relative to the region aspect ratio, adjust height
            scaled_height = h
            scaled_width = scaled_height / text_aspect_ratio
        else:
            # Text is wider relative to the region aspect ratio, adjust width
            scaled_width = w
            scaled_height = scaled_width * text_aspect_ratio

        # Center the found region within the original [left, top, right, bottom]
        find_left = left + (w - scaled_width) / 2
        find_top = top + (h - scaled_height) / 2
        find_right = find_left + scaled_width
        find_bottom = find_top + scaled_height

        return int(find_left), int(find_top), int(find_right), int(find_bottom)

    # If aspect ratio is close enough, return the original region
    return int(left), int(top), int(right), int(bottom)




class TypoAttackPlanner:
    def __init__(self, som_image_folder=None, temperature=0.2, max_tokens=4095, top_p=0.1):
        """
        Initialize the TypoAttackPlanner class.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.som_image_folder = som_image_folder

        self.diffuser = TextDiffuser()

        # system instruction
        with open('prompt/attack_step_give_answer_combine.txt',
                  'r') as file:
            self.instruction_combine = file.read()

        with open(
                'prompt/attack_adjust_plan.txt',
                'r') as file:
            self.instruction_adjust_plan = file.read()

    def attack(self, image_path, question, correct_answer):
        """
        Applies a 'typo attack' on the input PIL image and returns the modified image.

        Returns:
        The modified image with the applied 'typo attack'.
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Load som image and mask
        image_name = image_path.split("/")[-1]

        seg_image = Image.open(os.path.join(self.som_image_folder, image_name)).convert("RGB")
        mask = np.load(os.path.join(self.som_image_folder, image_name.replace(".jpg", ".npy")), allow_pickle=True)

        # get typo attack plan from chatgpt
        base64_image = pil_to_base64(image)
        base64_image_som = pil_to_base64(seg_image)
        # gpt-4o-2024-08-06

        completion_request = CompletionRequest(model="gpt-4o-2024-08-06", temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p,
                                               response_format=PlanSom)
        completion_request.set_system_instruction(self.instruction_combine)
        user_text = f"Image 0 is the original image, image 1 is the corresponding segmentation map. Observe the image and the corresponding segmentation map carefully. Question to attack: {question}. Correct answer: {correct_answer}. Please provide a detailed, step-by-step plan for achieving this goal."
        completion_request.add_user_message(text=user_text, base64_image=[base64_image, base64_image_som],
                                            image_first=True)
        completion = completion_request.get_completion_payload()
        plan_detail = completion.choices[0].message.parsed

        print("plan_detail:")
        print("image_analysis:", plan_detail.image_analysis)
        print("correct_answer:", plan_detail.correct_answer)
        print("incorrect_answer:", plan_detail.incorrect_answer)
        print("adversarial_text:", plan_detail.adversarial_text)
        print("text_placement:", plan_detail.text_placement)
        print("text_position_number:", plan_detail.text_position_number)
        print("short_caption_with_adversarial_text:", plan_detail.short_caption_with_adversarial_text)

        # add assistant message
        completion_request.add_assistant_message(text=f"{plan_detail}")
        plan_detail_origin = plan_detail.copy()

        # adjust plan to avoid region is the question target region
        user_text = self.instruction_adjust_plan

        completion_request.set_response_format(PlanSomAdjust)
        completion_request.add_user_message(text=user_text)
        completion = completion_request.get_completion_payload()
        plan_adjust = completion.choices[0].message.parsed

        plan_detail = plan_adjust.adjust_plan
        explanation = plan_adjust.adjust_explanation
        print("explanation:", explanation)
        print("plan_detail:")
        print("image_analysis:", plan_detail.image_analysis)
        print("correct_answer:", plan_detail.correct_answer)
        print("incorrect_answer:", plan_detail.incorrect_answer)
        print("adversarial_text:", plan_detail.adversarial_text)
        print("text_placement:", plan_detail.text_placement)
        print("text_position_number:", plan_detail.text_position_number)
        print("short_caption_with_adversarial_text:", plan_detail.short_caption_with_adversarial_text)

        # get the rectangle to place the text
        # if plan_detail.text_position_number is number and the number is in the mask
        if int(plan_detail.text_position_number) <= len(mask):
            target_mask = mask[int(plan_detail.text_position_number) - 1]['segmentation']
        else:
            print("text_position_number is out of range, use the largest mask")
            target_mask = mask[0]['segmentation']
        # target_mask = mask[0]['segmentation']
        label = True
        # target_mask = target_mask.T
        x, y, w, h = largest_inscribed_rectangle(target_mask, label)
        print("rectangle [x, y, w, h]:", [x, y, w, h])
        # change coordinate (0,0) from right-bottom to left-top, left to right is 0-1, top to bottom is 0-1
        mask_width, mask_height = target_mask.T.shape

        left, top, right, bottom = x / mask_width * image.width, y / mask_height * image.height, (
                x + w) / mask_width * image.width, (y + h) / mask_height * image.height

        print("rectangle [(left, top), (right, bottom)]:", [(int(left), int(top)), (int(right), int(bottom))])

        # resize by scale
        left, top, right, bottom = find_text_region(plan_detail.adversarial_text, left, top, right, bottom,
                                                    font_path="./fonts/arialbd.ttf",
                                                    font_size=20, aspect_ratio_threshold=0.1)
        print("Resized rectangle [(left, top), (right, bottom)]:", [(int(left), int(top)), (int(right), int(bottom))])


        # diffusion
        two_point_positions = [(int(left), int(top)), (int(right), int(bottom))]

        diffusion_result = self.diffuser.generate(two_point_positions, image_path, plan_detail.adversarial_text,
                                                  plan_detail.short_caption_with_adversarial_text, radio="Two Points",
                                                  scale_factor=2, regional_diffusion=True)


        diffusion_images = diffusion_result[0]
        diffusion_images = [diffusion_image.resize((image.width, image.height)) for diffusion_image in diffusion_images]

        return diffusion_images, seg_image, plan_detail_origin, plan_detail

