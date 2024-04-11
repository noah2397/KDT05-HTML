import cgi, sys, codecs, cgitb
import os
import torch
cgitb.enable()
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CNN_MODEL import *

# WEB 인코딩 설정
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


# 사용자 정의 함수
def print_browser(result=""): 
    filename='../ex_input.html' # 이 파일 기준 
    with open(filename, 'r', encoding='utf-8') as f:
        # HTML Header
        print("Content-Type: text/html; charset=utf-8;")
        print()
        # HTML Body
        print(f.read().format(result))
    
# 요청 및 브라우징(웹 페이지의 form태그 내의 input태그 입력값 가져와서 저장하고 있는 인스턴스 )
form = cgi.FieldStorage()
result=""
if 'img_file' in form:
    result = form.getvalue('img_file')

print_browser(result=result)



# model = torch.load("Bekki.pth")
# filepath="./bekki/51sZxP09G4L.jpg"
# anya_bekki_classification(model, filepath)