import requests
headers={'Content-Type':'application/json'}
data='{"question": "how to create a web application with python using front and desined with html and javascript"}'
#data='{"question": "how to open a file using pandas and python and extract timestamp from date column"}'
url='http://127.0.0.1:5000/predict'

response=requests.post(url,headers=headers,data=data)

print(response.text)