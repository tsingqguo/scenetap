from transformers import pipeline


class LingoJudge:
    def __init__(self, model_name='wayveai/Lingo-Judge', device=-1):
        """
        Initialize the Judge with a specified model.

        :param model_name: The name of the model to use in the pipeline
        :param device: The device to use (-1 for CPU, 0 for GPU)
        """
        self.model_name = model_name
        self.pipe = pipeline("text-classification", model=self.model_name, device=device)

    def evaluate(self, question, answer, prediction):
        """
        Evaluate the student's response to the question.

        :param question: The question posed
        :param answer: The correct answer
        :param prediction: The student's answer
        :return: A tuple (is_correct: bool, score: float)
        """
        input_text = f"[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"
        result = self.pipe(input_text)
        score = result[0]['score']
        return score > 0.5, score
