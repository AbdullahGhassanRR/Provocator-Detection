# Provocator-Detection
Given social media comment dataset, system could exploit who potentially provoke/initiate a bad sentiment for certain topics. This system uses both text classification and topic modelling to detect the provocator. 

Several files for the system included:
1. experiment.ipynb : A file where experiment (such as exploratory data analysis, data preprocessing, modelling, model evaluating, algorithm testing) is conducted.
2. backend.py : A FastAPI backend to run logical provocator detection algorithm.
3. frontend.py: A Streamlit frontent for user interface and receive output result from backend.
4. demo_data_final.csv : A dummy indonesian social media comment data example with appropriate columns for the system input. Data are collected from kaggle with details:
   a. text column collected from https://www.kaggle.com/datasets/alvinhanafie/dataset-for-indonesian-sentiment-analysis?select=train_preprocess_ori.tsv
   b. username column collected from https://www.kaggle.com/datasets/dionisiusdh/covid19-indonesian-twitter-sentiment
   c. with dummy date
