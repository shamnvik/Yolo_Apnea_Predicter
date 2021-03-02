from sklearn.metrics import f1_score
from .apneas import ApneaType
import numpy as np
import pandas as pd

class Evaluate:

    def __init__(self,predictions,ground_truth,apnea_types):
        print("init evaluate")
        self.predictions = predictions
        self.predictionsBool = predictions>0
        self.ground_truth = np.isin(ground_truth,[apnea.value for apnea in apnea_types])

    @property
    def metrics(self):
        raise NotImplementedError("evaluate.metrics not implemented")

    @property
    def f1(self):
        return f1_score(self.ground_truth,self.predictionsBool)

    @property
    def predictionAHI(self):
        df = self.get_predictions_as_df(self.predictions)
        return len(df["start"]) / (len(self.predictions) / (60 * 10)) * 60

    @property
    def groundTruthAHI(self):
        df = self.get_predictions_as_df(self.ground_truth)
        return len(df["start"]) / (len(self.ground_truth) / (60 * 10)) * 60

    def get_predictions_as_df(self, predictions):
        # with open('predictions_whole_recording.npy', 'wb') as f:
        #     np.save(f,self.predictions)

        # Taken from https://stackoverflow.com/questions/49491011/python-how-to-find-event-occurences-in-data
        indicators = (predictions > 0.0).astype(int)
        indicators_diff = np.concatenate([[0], indicators[1:] - indicators[:-1]])
        diff_locations = np.where(indicators_diff != 0)[0]

        print(diff_locations)
        assert len(diff_locations) % 2 == 0

        starts = diff_locations[0::2]
        ends = diff_locations[1::2]

        df = pd.DataFrame({'start': starts,
                           'end': ends, })

        df['min_confidence'] = [predictions[start:end].min() for start, end in zip(df["start"], df["end"])]
        df['max_confidence'] = [predictions[start:end].max() for start, end in zip(df["start"], df["end"])]
        df['duration'] = df["end"] - df["start"]
        return df


###
    #
    # def get_evaluation_metrics(self):
    #     annotation_metrics = {}
    #
    #     metric_end = int(float(max(self.ground_truth_length, self.last_predicted_index)))
    #     predictions = self.predictions[:metric_end]
    #     ground_truth = self.ground_truth[:metric_end]
    #     ground_truth_binary = np.ravel(binarize(ground_truth.reshape(1, -1), threshold=0))
    #     predictions_binary = np.ravel(binarize(predictions.reshape(1, -1), threshold=0))
    #
    #     annotation_metrics["accuracy_score"] = accuracy_score(ground_truth_binary, predictions_binary)
    #     annotation_metrics["f1_score"] = f1_score(ground_truth_binary, predictions_binary)
    #     annotation_metrics["precision_score"] = precision_score(ground_truth_binary, predictions_binary)
    #     annotation_metrics["recall_score"] = recall_score(ground_truth_binary, predictions_binary)
    #
    #     fpr, tpr, threshold = roc_curve(ground_truth.astype(bool), predictions)
    #     annotation_metrics["roc"] = {"fpr": fpr, "tpr": tpr, "treshold": threshold}
    #
    #     return annotation_metrics
    #
    # def plot_roc(self):
    #
    #     print(self.ground_truth_length)
    #     print(self.last_predicted_index)
    #
    #     metric_end = int(float(max(self.ground_truth_length, self.last_predicted_index)))
    #     predictions = self.predictions[:metric_end]
    #     ground_truth = self.ground_truth[:metric_end]
    #
    #     ground_truth[ground_truth > 1] = 0  # Filters Hypopnea|Hypopnea out as the model has only been training on OSA
    #
    #     fpr, tpr, threshold = roc_curve(ground_truth.astype(bool), predictions)
    #     roc_auc = auc(fpr, tpr)
    #     plt.title('Receiver Operating Characteristic')
    #     plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    #     plt.legend(loc='lower right')
    #     plt.plot([0, 1], [0, 1], 'r--')
    #     plt.xlim([0, 1])
    #     plt.ylim([0, 1])
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate')
    #     plt.show()
