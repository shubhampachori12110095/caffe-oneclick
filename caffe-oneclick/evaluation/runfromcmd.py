import subprocess
import os

imgdir="cnn"
files=os.listdir(imgdir)
cmdprefix="../bin/classification "+"modeldef/deploy.prototxt "+"trainedmodels/platerecognition_iter_5000.caffemodel "+"preprocess/mean.binaryproto "+"preprocess/category.txt "
files.sort(key= lambda x:int(x[:-4]))
for file in files:
    cmd=cmdprefix+imgdir+"/"+file
    subprocess.call(cmd)