import cgi, sys, codecs
import torch
from model import LinearModel 
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def print_browser(result=""): 
    filename='./cgi-bin/index.html'
    with open(filename, 'r', encoding='utf-8') as f:
        # HTML Header
        print("Content-Type: text/html; charset=utf-8;")
        print()
        # HTML Body
        print(f.read().format(result))
    
# 요청 및 브라우징
form = cgi.FieldStorage()
result=""

if 'crim' in form and 'zn' in form and 'indus' in form and 'chas' in form and 'nox' in form and 'rm' in form and 'age' in form and 'dis' in form and 'rad' in form and 'tax' in form and 'ptratio' in form and 'b' in form and 'lstat' in form:
    input_list = [float(form['crim'].value), float(form['zn'].value), float(form['indus'].value), float(form['chas'].value), float(form['nox'].value), float(form['rm'].value), float(form['age'].value), float(form['dis'].value), float(form['rad'].value), float(form['tax'].value), float(form['ptratio'].value), float(form['b'].value), float(form['lstat'].value)]
    input_tensor = torch.tensor(input_list, dtype=torch.float)
    model = torch.load("./optim.pth")
    prediction = model(input_tensor)
    result=prediction.item()

print_browser(result=result)