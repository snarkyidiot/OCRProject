# Hyper Speed OCR for medical applications
This project aims to develop a high-speed Optical Character Recognition (OCR) system tailored for ICU monitor screens, focusing on rapid and accurate recognition of digits. The objective is to enable real-time monitoring of vital signs, enhancing patient care in critical care settings.
The speed of the final model **300** micro seconds with an accuracy of 98.6%
## Datasets
Project consists of 4 main datasets
##### `Contours`
This contains [contours]("https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html") corresponding to each digit 
##### `core.v8i.tensorflow`
This is ther original Roboflow Data set version which we preprocessed
##### `Number_Dataset`
This is an unlabbeled Dataset which was made by crpping and preprocessing `core.v8i.tensorflow`
##### `dataset_of_screen`
This is a labelled dataset classified using a slow yet highly accurate OCR after seperating the digits from `Number_Dataset`
##### `moments.csv`
This File contains moments of the Shape of the Digits
##### `labels.csv`
This File contains labels of the Shape of the Digits
## Training/Exploratory Notebooks
##### `Preprocess.ipynb`
This Notebook was used for exploring various types of preprocessing and feature extraction for this purpose
##### `Train.ipynb`
This is the training script for the Random Forest and Hue Shape Matching Algorithms
## Utils
##### `models.pkl`
Final Weights of the Random Forest Classifier
##### `app.py`
Final user interface for the project
## How to run `app.py`
Requires python 3.10 or above

`python -m pip install requirements.txt`

`python app.py`

##### Training script
`Train.ipynb` can be used to modify the training script
