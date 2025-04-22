def is_correct_answer(answer, correct_answer, question, dataset_name, lingo_judge=None):
    if "typo_base" in dataset_name:
        options = question.split('(a) ')[1].split(' (b) ')
        option_a = options[0].lower()
        option_b = options[1].rstrip(' (c)').lower()
        if correct_answer == 'a':
            if answer == 'a' or "(a)" in answer or "a)" in answer or option_a in answer:
                return True
        elif correct_answer == 'b':
            if answer == 'b' or "(b)" in answer or "b)" in answer or option_b in answer:
                return True
        return False
    elif "vqav2" in dataset_name:
        return correct_answer.lower() in answer.lower()
    elif dataset_name == "LingoQA":
        is_correct, score = lingo_judge.evaluate(question, correct_answer, answer)
        return is_correct
