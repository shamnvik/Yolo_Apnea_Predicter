from setuptools import setup, find_packages

setup(
    name='yolo_apnea_predictor',
    version='0.1.2',
    package_dir={'': 'yolo_apnea_predicter'},
    packages=find_packages(),
    description='An interface for accessing predictions of apneas using abdo or thor sensors and yolo'

)
