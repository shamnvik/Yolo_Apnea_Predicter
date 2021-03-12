from sklearn.metrics import f1_score,recall_score
from sklearn.metrics import confusion_matrix,roc_curve,auc,accuracy_score,precision_score

import matplotlib.pyplot as plt

from .apneas import ApneaType
import numpy as np
import pandas as pd

class Evaluate:

    def __init__(self,predictions,ground_truth,apnea_types,threshold):
        self.predictions = predictions
        self.predictionsBool = predictions>threshold
        self.ground_truth = np.isin(ground_truth,[apnea.value for apnea in apnea_types])
        self.threshold = 0.25

    @property
    def scores(self):
        return {
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "roc_curve" : self.roc_curve,
            "auc": self.auc,
            "accuracy": self.accuracy,
            "ahi":self.ahi,
            "precision":self.precision,
            "recall":self.recall
        }

    @property
    def f1(self):
        return f1_score(self.ground_truth,self.predictionsBool)

    @property
    def confusion_matrix(self):

        tn, fp, fn, tp = confusion_matrix(self.ground_truth,self.predictionsBool).ravel()
        return {
            "tn":tn,
            "fp":fp,
            "fn":fn,
            "tp":tp
        }

    @property
    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.ground_truth, self.predictions,pos_label=1)
        return {
            "fpr":fpr,
            "tpr":tpr,
            "tresholds":thresholds
        }

    @property
    def auc(self):
        roc_curve = self.roc_curve
        fpr = roc_curve["fpr"]
        tpr = roc_curve["tpr"]
        return auc(fpr,tpr)


    @property
    def accuracy(self):
        return accuracy_score(y_true=self.ground_truth,y_pred=self.predictionsBool)

    @property
    def ahi(self):
        return {
            "prediction_AHI": self.predictionAHI,
            "true_AHI":self.groundTruthAHI
        }

    @property
    def precision(self):
        return precision_score(self.ground_truth,self.predictionsBool)

    @property
    def recall(self):
        return recall_score(self.ground_truth,self.predictionsBool)

    @property
    def predictionAHI(self):
        df = self.get_predictions_as_df(self.predictions)
        return len(df["start"]) / (len(self.predictions) / (60 * 10)) * 60

    @property
    def groundTruthAHI(self):
        df = self.get_predictions_as_df(self.ground_truth)
        return len(df["start"]) / (len(self.ground_truth) / (60 * 10)) * 60

    def get_predictions_as_df(self, predictions):

        indicators = (predictions > self.threshold).astype(int)

        in_event = False
        starts = []
        ends = []

        for i,val in enumerate(indicators):
            if val == True and not in_event:
                starts.append(i)
                in_event = True
            elif val == False and in_event:
                ends.append(i)
                in_event = False

        if in_event:
            ends.append(len(indicators))

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
