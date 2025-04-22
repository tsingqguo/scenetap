import argparse
import json
import logging
import os
import time

import torch
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset

from utils.completion_request import CompletionRequest
from utils.lingo_judge import LingoJudge
from utils.typo_attack_planner import TypoAttackPlanner, pil_to_base64
from utils.typo_attack_planner import format_instance_json
from utils.utils import is_correct_answer


class TypoDataset(Dataset):
    def __init__(self, question_path):
        # Question file
        with open(question_path, 'r') as f:
            self.questions = json.load(f)

    def __getitem__(self, index):
        line = self.questions[index]
        return line

    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, batch_size=1, num_workers=0):
    dataset = TypoDataset(questions)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name: gpt-4o")
    parser.add_argument("--dataset", type=str, default="typo_base_complex",
                        help="Dataset name: typo_base_complex, typo_base_color, vqav2_val2014")
    parser.add_argument("--attack", type=str, default="SceneTAP", help="Attack type: SceneTAP")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--log_dir", type=str, default="./log")

    # chatgpt
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=4095)
    parser.add_argument("--top_p", type=float, default=0)

    # som
    parser.add_argument("--slider", type=float, default=2)
    parser.add_argument("--filter", type=float, default=None)

    # seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # seed everything
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # logger
    args.log_dir = os.path.join(args.log_dir, args.model, args.dataset, args.attack)
    if args.attack == "SceneTAP":
        args.log_dir = os.path.join(args.log_dir, f"slider_{args.slider}", f"filter_{args.filter}")

    args.log_dir = os.path.join(args.log_dir, f"seed_{args.seed}")
    os.makedirs(os.path.dirname(os.path.join(args.log_dir, "log.txt")), exist_ok=True)
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(args.log_dir, "log.txt")
    test_log = logging.FileHandler(f'{log_path}', 'a', encoding='utf-8')
    test_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    test_log.setFormatter(formatter)
    logger.addHandler(test_log)

    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    KZT.setFormatter(formatter)
    logger.addHandler(KZT)

    logger.info("Config:")
    logger.info(json.dumps(args.__dict__, indent=2))
    logger.info("\n")

    # Path to save images
    image_save_dir = os.path.join(args.log_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)

    # Answer files
    answers_file = os.path.join(args.log_dir, "answer.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # judgement model
    if args.dataset == "LingoQA":
        lingo_judge = LingoJudge()
    else:
        lingo_judge = None

    # Attack planner
    if args.attack == "SceneTAP":
        som_base_path = "./som_images"
        som_image_folder = os.path.join(som_base_path, args.dataset, f"slider_{args.slider}", f"seed_{args.seed}", f"filter_{args.filter}")
        typo_attack_planner = TypoAttackPlanner(som_image_folder)

    # Load the questions
    with open(args.question_file, 'r') as f:
        questions = json.load(f)

    ans_file_list = []
    correct = 0
    total = 0
    for i, data in enumerate(questions):
        question_id = data["question_id"]
        image_name = data["image"]
        if ("vqav2" in args.dataset or "LingoQA" in args.dataset) and args.attack != "no_attack":
            image_name_save = f"{image_name.split('.')[0]}_{question_id}.{image_name.split('.')[1]}"
            if args.attack != "SceneTAP":
                image_name = f"{image_name.split('.')[0]}_{question_id}.{image_name.split('.')[1]}"
        else:
            image_name_save = image_name
        image_path = os.path.join(args.image_folder, image_name)

        question = data["text"]
        correct_answer = data["answer"]

        if args.attack == "SceneTAP":
            images, seg_image, plan_detail_origin, plan_detail = typo_attack_planner.attack(
                image_path, question, correct_answer)
            image = images[0]


            image.save(os.path.join(image_save_dir, f"{image_name_save}"))
            seg_image.save(os.path.join(image_save_dir, f"{image_name_save.replace('.jpg', '_seg.jpg')}"))
            # diffusion save path
            image_save_dir_diffusion = os.path.join(args.log_dir, "diffusion",
                                                    f"{image_name_save.replace('.jpg', '')}")
            os.makedirs(image_save_dir_diffusion, exist_ok=True)
            for k, img in enumerate(images):
                img.save(os.path.join(image_save_dir_diffusion, f"{k}.jpg"))
        else:
            image = Image.open(image_path).convert("RGB")
            images = [image]

        # Get the answer
        output_list = []
        judge_list = []
        for image in images:
            base64_image = pil_to_base64(image)
            completion_request = CompletionRequest(model=args.model, temperature=args.temperature,
                                                   max_tokens=args.max_tokens, top_p=args.top_p)
            completion_request.add_user_message_test(text=question, base64_image=[base64_image], image_first=True, detail="auto")
            completion = completion_request.get_completion_payload()
            answer = completion.choices[0].message.content

            answer = answer.lower()
            output_list.append(answer)
            judge_list.append(is_correct_answer(answer, correct_answer, question, args.dataset, lingo_judge=lingo_judge))

        # Save the answer
        if not(False in judge_list):
            correct += 1
            log_data = {"question_id": question_id,
                        "image": data["image"],
                        "text": question,
                        "outputs": output_list,
                        "answer": correct_answer,
                        "plan_detail_origin": format_instance_json(
                            plan_detail_origin) if args.attack == "SceneTAP" else None,
                        "plan_detail": format_instance_json(plan_detail) if args.attack == "SceneTAP" else None,
                        "judge_list": judge_list,
                        "is_correct": True
                        }
            ans_file_list.append(log_data)
            logger.info(log_data)

        else:
            log_data = {"question_id": question_id,
                        "image": data["image"],
                        "text": question,
                        "outputs": output_list,
                        "answer": correct_answer,
                        "plan_detail_origin": format_instance_json(
                            plan_detail_origin) if args.attack == "SceneTAP" else None,
                        "plan_detail": format_instance_json(plan_detail) if args.attack == "SceneTAP" else None,
                        "judge_list": judge_list,
                        "is_correct": False
                        }
            ans_file_list.append(log_data)
            logger.info(log_data)
        total += 1

    ans_file.write(json.dumps(ans_file_list, indent=2))
    ans_file.close()
    logger.info(f"Correct: {correct}/{total}")
    logger.info(f"Accuracy: {correct / total}")
    logger.info(f"ASR: {(total - correct) / total}")
