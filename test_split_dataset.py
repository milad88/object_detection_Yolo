from os import listdir
from os.path import isfile, join

mypath1 = './data/images/train'
mypath2 = './data/labels/train'
images = sorted([f.split('.jpg')[0] for f in listdir(mypath1) if isfile(join(mypath1, f))])
labels = sorted([f.split('.txt')[0] for f in listdir(mypath2) if isfile(join(mypath2, f))])

print("something")