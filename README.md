# Yolo_Apnea_Predicter
Master thesis code for predicting apnea events using yolo.

###NB! Still a work in progress, expect interface to change

# Description

Predict Apnea events on ABDO signal from .edf file or other numpy array signal using trainled Yolo model.
Outputs nsrr-xml info, but wil later be expanded to be able to compare to true signals annotated by sleep technicians,
and will return values of how good the predictions are.

When more models have been generated and trained, the intention is that this repo will handle them as well by changing
paramenters when initializing the detector
# Usage:


Predict Apnea events on .edf file 
# Usage:


```bash
usage: main.py [-h] file

```
# Arguments

|short|long|default|help|
| :---: | :---: | :---: | :---: |
|`-h`|`--help`||show this help message and exit|
