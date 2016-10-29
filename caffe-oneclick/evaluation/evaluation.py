import os
import sys
sys.path.append(r'../../python')
import caffe
import numpy as np
import time
import shutil

datadir='../data'
modeldef='../modeldef/deploy.prototxt'
pretrainedmodel='../trainedmodels/platerecognition_iter_12000.caffemodel'
meanfile='../preprocess/mean.npy'
categoryfile='../preprocess/synset_words.txt'
errordir='../error'

def getclassifier():
    image_dims = [20,20]
    channel_swap=[2,1,0]
    raw_scale=255.0
    mean=np.load(meanfile)
    classifier = caffe.Classifier(modeldef, pretrainedmodel,
            image_dims=image_dims, mean=mean,raw_scale=raw_scale,
            channel_swap=channel_swap)
    caffe.set_mode_gpu()
    return classifier

def evaluation():
    classifier=getclassifier()
    f=open(categoryfile)
    words=f.readlines()
    f.close()
    start = time.time()
    if not os.path.exists(errordir):
        os.mkdir(errordir)
    subdirs=os.listdir(datadir)
    datacount=0
    errorcount=0
    for subdir in subdirs:
        files=os.listdir(datadir+'/'+subdir)
        print subdir+":"
        for file in files:
            datacount=datacount+1
            imgpath=datadir+'/'+subdir+'/'+file
            inputs = [caffe.io.load_image(imgpath)]
            try:
                predictions = classifier.predict(inputs)
            except Exception as e:
                print e
            p=predictions[0,:].argmax()
            w=words[p].split()[1]
            if subdir!=w:
                print subdir,w
                errorcount=errorcount+1
                if not os.path.exists(errordir+'/'+subdir):
                    os.mkdir(errordir+'/'+subdir)
                errorfilepath=errordir+'/'+subdir+'/'+file[:-4]+"_"+subdir+'_'+w+'.jpg'
                shutil.copy(imgpath,errorfilepath)
    print("Done in %.2f s." % (time.time() - start))
    print errorcount,datacount
    print errorcount*100/datacount
if __name__=='__main__':
    evaluation()
