import cgi, sys, codecs
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from NLP_MODEL import * 


# WEB 인코딩 설정
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())



# 사용자 정의 함수
def print_browser(result=""): 
    filename='./html/test.html'
    with open(filename, 'r', encoding='utf-8') as f:
        # HTML Header
        print("Content-Type: text/html; charset=utf-8;")
        print()
        # HTML Body
        print(f.read().format(result))
    
# 요청 및 브라우징
form = cgi.FieldStorage()
result=""
if 'data1' in form and 'data2' in form:
    result = form.getvalue('data1') + " 너무너무 " + form.getvalue('data2')
    MODEL = torch.load("model.pth")  
    result = predict(MODEL, result)
    print(result)

print_browser(result=result)