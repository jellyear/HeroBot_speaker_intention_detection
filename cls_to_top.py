import re

clsfile = './data/tags.cls'
topfile = './data/tags.top'

with open(clsfile) as rf:
    list= [line.rstrip() for line in rf.readlines()]

with open(topfile, 'w', encoding='utf-8') as wf:
    for line in list:
        if(line==''): continue
        value= line
        name= re.sub('[-=.,#/?!:$}]', '', line)
        name= name.replace(' ','')
        wf.write("concept: ~"+name+" MORE NOUN [ " + value +" ]\n")