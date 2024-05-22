
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/model_nb.pkl", 'rb') as f:
    model = pickle.load(f)

def predict_text(text):
    
    text_argmax = model.predict([text])

    class_names = ['Common Cold', 
                   'Dengue', 
                   'Malaria', 
                   'Typhoid']
    
    prediction = class_names[text_argmax[0]]
    return prediction