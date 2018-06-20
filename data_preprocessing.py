#-*- coding:utf-8 -*-
import urllib3
import json

# using morpheme analysis api 
# http://aiopen.etri.re.kr/doc_language.php
openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
#accessKey = "c40eadfc-a128-4037-9bbb-73ba36399267"
accessKey = "386fcdc1-f5b8-4914-9774-9d964521d2a5"
analysisCode = "morp"

txtfile = ['./data/classification.txt']
trainfile = ['./data/classification.train']
tagfile = './data/tags.cls'

for i in range(len(txtfile)):
    with open(txtfile[i],'r',encoding='utf-8') as rf:
        datas= [line.strip() for line in rf.readlines()]
        
    with open(trainfile[i],'w',encoding='utf-8') as wf, open(tagfile, 'w', encoding='utf-8') as tf:
        tag = ''
        for data in datas :
            if data=='' : continue
            # Extract tag
            elif data.startswith('===') :
                tag = data[3:].title()
                tf.write(tag+"\n")
            # Morpheme analysis
            else :
                requestJson = {
                    "access_key": accessKey,
                    "argument": {
                        "text": data,
                        "analysis_code": analysisCode
                    }
                }

                http = urllib3.PoolManager()
                response = http.request(
                    "POST",
                    openApiURL,
                    headers={"Content-Type": "application/json; charset=UTF-8"},
                    body=json.dumps(requestJson)
                )
                string = response.data.decode('utf-8')
                json_obj = json.loads(string)

                # After morpheme analysis           
                morpdic = json_obj['return_object']['sentence'][0]['morp']
                morptxt = ''
                for dic in morpdic :
                    if dic['type'].startswith('J') == False and dic['type'].startswith('E') == False and dic['type']!='SF' and dic['type']!= 'VCP': 
                        morptxt+= dic['lemma']+" " 
                wf.write(morptxt.strip()+","+tag+"\n")