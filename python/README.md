# Aliran - Genre Classification

Making use of Machine learning techniques such as kNN (K-Nearest Neighbors) to detect music genres from audio recordings.

## Dataset
The dataset is not present, however can be download from: http://marsyas.info/downloads/datasets.html

This dataset is appr. 1.2GB in size.

This is a fairly small dataset and for future testing the Million Song Dataset could also be used (300GB+): http://millionsongdataset.com

## Installation
1. Download this repository
2. Download the Dataset as mentioned in the `Dataset` section.
3. Drag the `Data` directory into this project (or create a `Data` directory and drag all files inside that directory).
4. In `genre_classifier.py` in the `extract_features` method, ensure the `directory` is set correctly.
So if you're structure is `my_project/Data/genres_original/blues/blues_01.wav` then `directory` would be `Data/genres_original/`. (make sure you end with a `/`).
5. If it's the first time running ensure that `extracting` is set the `True` in `genre_classifier.py` under `__main__`.
This will ensure that a model is going to be build. 
You can set this back to `False` once a model is build.
6. Run `python3 genre_classifier.py` and it should start to spit out values in your Terminal.
It's finished once you'll receive a message saying `Prediction Accuracy is:`

