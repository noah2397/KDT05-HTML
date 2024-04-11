import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def anya_bekki_classification(model, filepath):
    img=cv2.imread(filepath)
    img = cv2.resize(img, (50,50))
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    if not model(torch.FloatTensor(img.reshape(1,3,50,50))).argmax() :
        return "Anya~"
    else : 
        return "Bekki~"
        
        
if __name__ == "__main__":
    model = torch.load("Bekki.pth")
    filepath="./bekki/51sZxP09G4L.jpg"
    anya_bekki_classification(model, filepath)