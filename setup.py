from setuptools import setup, find_packages

setup(
    name='yolo_apnea_predicter',
    version='0.0.1',
    packages=find_packages(include=['src', 'src.*']),
    description='An interface for accessing predictions of apneas using abdo or thor sensors and yolo'

)
