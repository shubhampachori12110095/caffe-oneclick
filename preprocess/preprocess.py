import os,argparse,random,shutil

def sampledata(chardir="E:/PatternRecognition/EasyPR/EasyPR-1.5beta/resources/train/ann"):
    numchars=10
    subdirs=os.listdir(chardir)
    for i in range(numchars):
        subdir=chardir+"/"+subdirs[i]
        files=os.listdir(subdir)
        dstdir="../data/"+subdirs[i]
        if not os.path.exists(dstdir):
            os.mkdir(dstdir)
        for file in files:
            srcpath=subdir+"/"+file
            dstpath=dstdir+"/"+file
            shutil.copy(srcpath,dstpath)

def main(args):
    datadir=args.rootdir+"/"+args.dataname
    print "loading data from "+datadir+":"
    trainfile=open("../preprocess/train.txt","w");
    valfile=open("../preprocess/val.txt","w");
    categoryfile=open("../modeldef/labels.txt",'w')
    paths=os.listdir(datadir)
    classindex=0
    trainpaths=[]
    valpaths=[]
    categorys=[]
    for subdir in paths:
        if(os.path.isdir(datadir+"/"+subdir)):
            print subdir
            categorys.append(str(classindex)+" "+subdir+"\n")
            files=os.listdir(datadir+"/"+subdir)
            random.shuffle(files)
            num2train=len(files)*args.trainrtaio
            for fileindex,file in enumerate(files):
                if fileindex<num2train:
                    trainpaths.append(args.dataname+"/"+subdir+"/"+file+" "+str(classindex)+"\n")
                else:
                    valpaths.append(args.dataname+"/"+subdir+"/"+file+" "+str(classindex)+"\n")
            classindex=classindex+1

    for category in categorys:
        categoryfile.write(category)

    random.shuffle(trainpaths)
    random.shuffle(valpaths)
    print "writing to files...:"
    for trainpath in trainpaths:
        trainfile.write(trainpath)

    for valpath in valpaths:
        valfile.write(valpath)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir",default="../",help="Directory of images to classify")
    parser.add_argument("--dataname",default="data",help="Dataset name")
    parser.add_argument("--trainrtaio",default=0.8,help="Train ratio ")
    parser.add_argument("--valrtaio",default=0.2,help="Val ratio")
    parser.add_argument("--testrtaio",default=0.1,help="Test ratoi")
    args = parser.parse_args()
    main(args)
    print "Done"