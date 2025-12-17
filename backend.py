import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from torch.cuda import empty_cache
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict
import sys
import torch
import gc
from bertopic import BERTopic

# -----------------------------
# Configuration & Model Loading
# -----------------------------

app = FastAPI(title="Provocator Detection API")

# Load tokenizer and model once at startup
MODEL_ID = "answerdotai/modernBERT-base"
CACHE_DIR = "C:/Users/Abdullah Ghassan/.cache/huggingface/hub"
MODEL_PATH = "best_model_modernBERT_2.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

# Load model classifier
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=3,
        cache_dir=CACHE_DIR
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# load llm
llm = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", device='cuda', dtype=torch.bfloat16, return_full_text = False)

class TensorFactory(TensorDataset):
    """Helper to create TensorDataset without labels"""
    def __init__(self, input_ids, attention_mask):
        super().__init__(input_ids, attention_mask)
        
def predict_labels(texts: List[str], batch_size: int = 16) -> List[int]:
    """
    input: list of texts
    process: run inference with classifier model
    output: return list of predicted labels
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=36,
        return_tensors='pt'
    )
    dataset = TensorFactory(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    del dataset, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return all_preds

def merge_sample_prediction(test:pd.DataFrame, predictions: List[int]) -> pd.DataFrame:
    """
    input: dataframe and list of predicted labels
    process: merge dataframe and predicted labels
    output: return a dtaframe which consist a predicted labels column
    """
    row_start = test.shape[0] - len(predictions) 
    df_temp = test.iloc[row_start:]
    df_temp = df_temp.reset_index()
    pred_temp = pd.Series(predictions, name = 'prediksi_model')
    result2 = pd.merge(df_temp, pred_temp, left_index = True, right_index = True, how = 'left')
    return result2
    
def sentiment_process(test: pd.DataFrame) -> pd.DataFrame:
    texts = test['clean_text'].astype(str).tolist()
    labels = predict_labels(texts)

    # merge_data == dataframe which consist the clean_text and the predicted label by model
    merge_data = merge_sample_prediction(test, labels)

    # index must be resetted and drop the garbage column from reset_index function ALWAYS
    merge_data.reset_index(inplace = True)
    merge_data.drop('level_0', axis = 1, inplace = True)
    
    return merge_data.loc[merge_data['prediksi_model'] == 1]


#
# TOPIC MODELLING
#

def merge_sample_topic(test:pd.DataFrame, predictions: List[int]) -> pd.DataFrame:
    """
    input: dataframe and list of predicted topic
    process: merge dataframe and predicted topic
    output: return a dtaframe which consist a predicted topic column
    """
    row_start = test.shape[0] - len(predictions) 
    df_temp = test.iloc[row_start:]
    pred_temp = pd.Series(predictions, name = 'topics')
    result2 = pd.merge(df_temp, pred_temp, left_index = True, right_index = True, how = 'left')
    return result2

def extract_topwords(topic_model):
    topic_info = topic_model.get_topic_info()
    topic_descriptions = {}
    for topic_id in topic_info['Topic']:
        if topic_id == -1: 
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)]
        topic_descriptions[topic_id] = ", ".join(words[:10])
    return topic_descriptions
    
def refine_topic_label(top_words: str) -> str:
    #remove empty/low-quality top_words before prompting
    terms = [t.strip() for t in top_words.split(",") if t.strip() and len(t) > 2]
    if not terms:
        return "General Topic"
    top_words_clean = ", ".join(terms[:8])

    # prompting
    prompt = ("You are an expert in topic labeling."
              "Given the following key terms from a topic model, summarize the topic in 2-4 words.\n"
              "Respond ONLY with the topic name. No explanations. No punctuation.\n\n"
              f"Key terms: {top_words_clean}\n\n"
              "Topic name:")
    try:
        response = llm(prompt, max_new_tokens=10, do_sample=False, return_full_text = False)
        return response[0]['generated_text'].strip()
    except Exception as e:
        return "Unknown Topic"


def topic_modelling_process(topic_modelling_input: pd.DataFrame) -> pd.DataFrame:
    """
    input: a dataframe after sentiment process (topic modelling input dataframe)
    process: 
    output: return a dataframe after topic modelling process
    """

     # index must be resetted and drop the garbage column from reset_index function ALWAYS

    topic_modelling_input.reset_index(inplace = True)
    topic_modelling_input.drop('level_0', axis = 1, inplace =True)
    topic_model = BERTopic.load("topic_model")   
    
    model_input = topic_modelling_input['clean_text']
    topics, probs = topic_model.transform(model_input)

    topic_modelling_output = merge_sample_topic(topic_modelling_input, topics)

    # Extract top words per topic
    topic_descriptions = extract_topwords(topic_model)

    del topic_model
    gc.collect()
    torch.cuda.empty_cache()

    # load llm to fine tune labels
    refined_labels = {}
    
    for topic_id, words in topic_descriptions.items():
        print(f"Refining topic {topic_id} with words: {words}")
        refined = refine_topic_label(words)
        print(f" → Got: {refined}")
        refined_labels[topic_id] = refined

    topic_modelling_output['word_topic'] = (
    topic_modelling_output['topics']
    .map(refined_labels)
    .fillna("Unknown Topic")
    .astype(str))

    return topic_modelling_output, refined_labels
    
# -----------------------------
# Pydanctic Configuration (to pack data for smooth transfer between frontend and backend)
# -----------------------------

class InputRecord(BaseModel):
    username: str
    date: str
    clean_text: str
    text: str
    # Add other fields if needed, but only these are used

class PredictionResult(BaseModel):
    username: str
    date: str
    clean_text: str
    topics: int
    word_topic: str

class TopicLabels(BaseModel):
    labels: Dict[str, str] 

class FinalResponse(BaseModel):
    predictions: List[PredictionResult]
    topic_labels: Dict[str, str]

    
# -----------------------------
# API Endpoint 
# -----------------------------

@app.post("/predict", response_model=FinalResponse)

#
# MAIN 
#
async def predict_provocators(inputs: List[InputRecord]):
    """
    Accepts list of records with 'username', 'date', 'clean_text'
    Returns list of {username, date, label}
    """
    if not inputs:
        raise HTTPException(status_code=400, detail="Empty input received")

    global model, tokenizer    

    try:
        # Convert Frontend input to DataFrame 
        test = pd.DataFrame([r.dict() for r in inputs])

        # sentiment process
        df_1 = sentiment_process(test)

        # clean gpu vram
        model.to('cpu')
        del tokenizer, model
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

        # topic modelling process
        df_2, refined_labels = topic_modelling_process(df_1)

        # df_2 and refined_labels converter to json
        df_2['topics'] = df_2['topics'].astype(int) # ensure topics is int
        predictions = df_2[["username", "date", "clean_text", "topics", "word_topic"]].to_dict(orient="records")
        topic_labels = {str(k): v for k, v in refined_labels.items()}

        return FinalResponse(
            predictions=predictions,
            topic_labels=topic_labels
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")