import json
import requests
from requests.auth import HTTPBasicAuth

username = "username"
password = "password"

# Base Service Communications URI
baseuri = "https://api.admin.microsoftonline.com/shdtenantcommunications.svc"
headers = {"accept": "application/json;odata=verbose"}
auth = {"username": username, "password": password}
# URI Paths
serviceinfo = "/GetServiceInformation"
register = "/Register"

payload = {'userName': username, 'password': password}
myheaders = {'Content-Type': 'application/json'}
data=json.dumps(payload)
response = requests.post(baseuri+register,data=json.dumps(payload),headers=myheaders)
responsedata = json.loads(response.text)
cookie = responsedata.get("RegistrationCookie")

payload1 = {'lastCookie':cookie,'locale':"en-US"} 
response = requests.post(baseuri+serviceinfo,data=json.dumps(payload1),headers=myheaders)
responsedata = json.loads(response.text)
for myobject in responsedata:
   print myobject.get("ServiceName")