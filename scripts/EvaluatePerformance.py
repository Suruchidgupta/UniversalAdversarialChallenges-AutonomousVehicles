from collections import defaultdict

class EvaluatePerformance(object):
    """
    This class computes the following scores for the Object Detection predictions:
    - Confusion Matrix elements
    - Precision
    - Recall
    - F1-score

    and collates the scores for all the folders
    """

    def compute_scores(self, expected, predicted):
        """
        This method computes confusion matrix, precision, recall and f1-score for the provided dataframes
        :param expected: groundtruth values
        :param predicted: predicted values
        :return: dict with the above given scores
        """
        scores_dict = defaultdict()
        true_positive = true_negative = false_positive = false_negative = 0
        # Setting index for Ground truth and removing all 0 columns
        expected.set_index('Image', inplace=True)

        for index, pred in predicted.iterrows():
            actual = expected.loc[self.alter_index(index)]

            # Collating columns with 0 values
            act_zero = actual[actual == 0]
            pred_zero = pred[pred == 0]
            common = list(set(act_zero.index) & set(pred_zero.index))

            # Deleting 0 value columns to create confusion matrix
            actual = actual.drop(common)
            pred = pred.drop(common)

            try:
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
            except:
                print(pred, actual, pred_value, act_value, index)
        # Adding the values for confusion matrix in the dictionary
        scores_dict['true_positive'] = true_positive
        scores_dict['true_negative'] = true_negative
        scores_dict['false_positive'] = false_positive
        scores_dict['false_negative'] = false_negative

        # Computing precision, recall and F1-scores
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive/ (true_positive + false_negative)
        f1_score = round((2 * precision * recall) / (precision + recall), 2)

        # Adding cumulative scores in the dictionary
        scores_dict['precision'] = precision
        scores_dict['recall'] = recall
        scores_dict['f1_score'] = f1_score

        return scores_dict

    def alter_index(self, index):
        """
        The index for ground truth and predictions is inconsistent.
        This method returns suitable index for ground truth.
        :param index: prediction index
        :return: groundtruth index
        """
        return index.split('/')[-1]