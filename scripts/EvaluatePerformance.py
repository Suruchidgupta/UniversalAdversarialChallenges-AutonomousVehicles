import pandas as pd
from collections import defaultdict


class EvaluatePerformance(object):
    """
    This class computes the following scores for the Object Detection predictions:
    - Confusion Matrix elements
    - Accuracy
    - Precision
    - Recall
    - F1-score
    and collates the scores for all the folders
    """

    def compute_scores(self, expected, predicted):
        """
        This method computes confusion matrix, precision, recall and f1-score for the provided dataframes
        :param expected: ground truth values
        :param predicted: predicted values
        :return: dict with the above given scores
        """
        scores_dict = defaultdict()
        true_positive = true_negative = false_positive = false_negative = 0

        for index, pred in predicted.iterrows():
            actual = expected.loc[self.ground_truth_index(index)]

            # Collating columns with 0 values
            act_zero = actual[actual == 0]
            pred_zero = pred[pred == 0]
            common = list(set(act_zero.index) & set(pred_zero.index))

            # Deleting 0 value columns to create confusion matrix
            actual = actual.drop(common)
            pred = pred.drop(common)

            # Computing confusion matrix for each label
            for label, pred_value in pred.items():
                act_value = actual[label]

                if pred_value < act_value:
                    # Detected less objects, missed objects contribute to FN
                    false_negative = false_negative + (act_value - pred_value)
                    true_positive = true_positive + pred_value
                elif pred_value > act_value:
                    # Detected extra objects, extra objects contribute to FP
                    false_positive = false_positive + (pred_value - act_value)
                    true_positive = true_positive + act_value
                elif pred_value == act_value and pred_value != 0:
                    # All object correctly classified
                    true_positive = true_positive + pred_value
                else:
                    # no object in frame to be detected
                    true_negative = true_negative + 1

        # Adding the values for confusion matrix in the dictionary
        scores_dict['true_positive'] = true_positive
        scores_dict['true_negative'] = true_negative
        scores_dict['false_positive'] = false_positive
        scores_dict['false_negative'] = false_negative

        # Computing accuracy, precision, recall and F1-scores
        (accuracy, precision, recall, f1_score) = self.compute_metrics(true_positive, true_negative,
                                                                       false_positive, false_negative)

        # Adding cumulative scores in the dictionary
        scores_dict['accuracy'] = accuracy
        scores_dict['precision'] = precision
        scores_dict['recall'] = recall
        scores_dict['f1_score'] = f1_score

        return scores_dict

    def ground_truth_index(self, index):
        """
        The index for ground truth and predictions is inconsistent.
        This method returns suitable index for ground truth.
        :param index: prediction index
        :return: ground truth index
        """
        return index.split('/')[-1]

    def compute_metrics(self, true_positive, true_negative, false_positive, false_negative):
        """
        The method computes and returns accuracy, precision, recall and f1-score from confusion matrix values
        :param true_positive: From confusion matrix
        :param true_negative: From confusion matrix
        :param false_positive: From confusion matrix
        :param false_negative: From confusion matrix
        :return: tuple containing accuracy, precision, recall and f1-score
        """
        accuracy = round((true_positive + true_negative) /
                         (true_positive + true_negative + false_negative + false_positive), 2)
        precision = round(true_positive / (true_positive + false_positive), 2)
        recall = round(true_positive / (true_positive + false_negative), 2)
        f1_score = round((2 * precision * recall) / (precision + recall), 2)

        return accuracy, precision, recall, f1_score

    def publish_results(self, scores):
        """
        The method formats the scores dictionary collates the scores for each category and prints on the console
        :param scores: dictionary of scores for each category
        :return: None
        """
        # Setting display option to view all columns
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.width', 500)

        # Printing metrics for each folder
        metrics = pd.DataFrame(scores).T
        metrics = metrics[['accuracy', 'precision', 'recall', 'f1_score',
                           'true_positive', 'true_negative', 'false_positive', 'false_negative']]

        # Updating the index values in the dataframe
        as_list = metrics.index.tolist()
        as_list = [idx.replace('../data/', '') for idx in as_list]
        metrics.index = as_list

        # Computing cumulative confusion matrix
        total_tp = metrics['true_positive'].sum()
        total_tn = metrics['true_negative'].sum()
        total_fp = metrics['false_positive'].sum()
        total_fn = metrics['false_negative'].sum()

        # Computing cumulative metrics, adding them to the metrics table
        (total_accuracy, total_precision, total_recall, total_f1_score) = self.compute_metrics(total_tp, total_tn,
                                                                                               total_fp, total_fn)
        metrics.loc['Total'] = [total_accuracy, total_precision, total_recall, total_f1_score, total_tp, total_tn,
                                total_fp, total_fn]

        # Printing final metrics
        print(metrics)
