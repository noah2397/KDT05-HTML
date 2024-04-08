import torch.nn as nn
import torch.nn.functional as F
import pickle
from konlpy.tag import Okt
import torch

class TextModel(nn.Module):
    def __init__(self, VOCAB_SIZE, EMBEDD_DIM, HIDDEN_SIZE, NUM_CLASS):
        super(TextModel, self).__init__()
        self.embedding = nn.EmbeddingBag(VOCAB_SIZE, EMBEDD_DIM, sparse=False)
        self.hidden = nn.Linear(EMBEDD_DIM, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASS)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.hidden.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        hidden_output = F.relu(self.hidden(embedded))
        output = self.fc(hidden_output)
        return output

def predict(MODEL, text):
    with torch.no_grad():
        with open('VOCAB', 'rb') as f:
            VOCAB = pickle.load(f)
        tokenizer = Okt()
        
        text_pipeline = lambda x: VOCAB(tokenizer.morphs(x))
        label_text = ['두한', '정진영', '이정재', '이승만']
        text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text = text.unsqueeze(0)
        offsets = None
        predicted_label = MODEL(text, offsets)
        print(f"예측 : {label_text[predicted_label.argmax(1).item()]}")

if __name__ == '__main__':
    MODEL = torch.load("model.pth")  
    text = '오늘 저녁 뭐먹죠'
    predict(MODEL, text)
