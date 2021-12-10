import numpy as np
from nltk.metrics.distance import edit_distance


def metrics(ground_truth: list, predictions: list, type: str):
    """ Metrics to evaluate quality of OCR models.

    Args:
        ground_truth (list): list of golden sentences as labels
        predictions (list): list of predicted sentences
        type (str): Character-based, Sentence-based accuracy and Normalized edit distant
    """
    if type == 'char_acc':
        char_accuracy: list = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    char_accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        char_accuracy.append(1)
                    else:
                        char_accuracy.append(0)
        accuracy = np.mean(np.array(char_accuracy).astype(np.float32), axis=0)

        return accuracy
    elif type == 'accuracy':
        correct_count: int = 0

        for pred, gt in zip(predictions, ground_truth):
            if pred == gt:
                correct_count += 1

        accuracy = correct_count / len(ground_truth)

        return accuracy
    elif type == 'normalized_ed':
        norm_ed: float = 0
        for pred, gt in zip(predictions, ground_truth):
            if len(gt) == 0 or len(pred) == 0:
                norm_ed += 0
            elif len(gt) > len(pred):
                norm_ed += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ed += 1 - edit_distance(pred, gt) / len(pred)

        return norm_ed / len(ground_truth)
    else:
        raise NotImplementedError('Other accuracy compute mode has not been implemented')
