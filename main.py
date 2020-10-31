import argparse

from yolo_apnea_predicter.apnea_detector import ApneaDetector


if __name__ == '__main__':
    print("Starting cli")
    parser = argparse.ArgumentParser()

    detector = ApneaDetector()
    #print(detector.get_new_predictions())

    parser = argparse.ArgumentParser(description='Predict Apnea events on .edf file ')
    parser.add_argument('file', help='path to a .edf file to analyze')
    parser.add_argument(
        "-x", '-xml', help='Output predictions annotations to xml file', action="store_true")
    args = parser.parse_args()
    print(args.file)