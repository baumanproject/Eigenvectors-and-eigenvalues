#%%
import os

def check(str):
    if(str in mydict.keys()):
        mydict[str] = mydict[str]+1
    else:
        mydict[str]=1


mydict = {}
dirname = os.path.dirname('test.txt')
for line in open(dirname):
    for str in line.split():
        check(str)
print(mydict)



#%%
