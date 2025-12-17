# Provocator-Detection
Given social media comment dataset, system could exploit who potentially provoke/initiate a bad sentiment for certain topics. This system uses both text classification and topic modelling to detect the provocator. 

Several files for the system included:
1. experiment.ipynb : A file where experiment (such as exploratory data analysis, data preprocessing, modelling, model evaluating, algorithm testing) is conducted.
2. backend.py : A FastAPI backend to run logical provocator detection algorithm.
3. frontend.py: A Streamlit frontent for user interface and receive output result from backend.
4. demo_data_final.csv : A dummy indonesian social media comment data example with appropriate columns for the system input. Data are collected from kaggle with details:
- text column collected from https://www.kaggle.com/datasets/alvinhanafie/dataset-for-indonesian-sentiment-analysis?select=train_preprocess_ori.tsv
- username column collected from https://www.kaggle.com/datasets/dionisiusdh/covid19-indonesian-twitter-sentiment
- with dummy date

# How to run locally with Anaconda
#### for backend:
0. open the first CLI (e.g. cmd, anaconda prompt) 
1. type cd path/to/downloaded/files
2. type conda activate <first_anaconda_envs> (must have uvicorn and all backend code dependencies)
3. type uvicorn backend:app --reload  
3.1. copy-paste or ctrl + click the given URl from CLI output
3.2. on the browser tab, type \docs on the endpoint (e.g. localhost:0000\docs)
4. klik API that want to interact with 
5. klik try it out
6. klik execute
7. copy request URL, continue to frontend

#### for frontend
8. paste request URL in file frontend.py on line where this code written: requests.post(url="<paste-here>", ...)
9. type cd path/to/downloaded/files
10. type conda activate <second_anaconda_envs> (due to dependencies conflict, frontend enviroment must be different from the backend. this frontend enviroment which have streamlit and all frontend code dependencies)
11. streamlit run frontend.py
12. copy-paste or ctrl + click the given URL from CLI output.

