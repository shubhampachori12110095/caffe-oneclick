import os
import random
import argparse

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%s\t' % item[1]
            line += '%d\n' % item[2]
            fout.write(line)

def list_image(root, recursive, exts):
    image_list = []
    if recursive:
        cat = {}
        for path, subdirs, files in os.walk(root, followlinks=True):
            subdirs.sort()
            print(len(cat), path)
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (len(image_list), root[3:]+"/"+os.path.relpath(fpath, root), cat[path])
    else:
        for fname in os.listdir(root):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (len(image_list), os.path.relpath(fpath, root), 0)
    
def make_list(args):
    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    print "processed "+ str(len(image_list))+"files"
    sep = int(len(image_list) * args.train_ratio)
    sep_test = int(len(image_list) * args.test_ratio)
    write_list("../preprocess/val.txt",image_list[:sep_test])
    write_list("../preprocess/train.txt",image_list[sep_test:sep_test + sep])

def generatechardatasets(datadir):
    datafile=open("preprocess/"+datadir+".txt","w");
    trainfile=open("preprocess/train.txt","w");
    testfile=open("preprocess/val.txt","w");
    categoryfile=open("preprocess/synset_words.txt",'w')
    paths=os.listdir(datadir)
    classindex=0
    for subdir in paths:
        if(os.path.isdir(datadir+"/"+subdir)):
            files=os.listdir(datadir+"/"+subdir)
            categoryfile.write(str(classindex)+"\t"+subdir+"\n")
            random.shuffle(files)
            num2train=len(files)*0.8
            fileindex=0
            for file in files:
                print datadir+"/"+subdir+"/"+file+"\t"+str(classindex)
                datafile.write(datadir+"/"+subdir+"/"+file+"\t"+str(classindex)+"\n")
                if fileindex<num2train:
                    trainfile.write(datadir+"/"+subdir+"/"+file+"\t"+str(classindex)+"\n")
                else:
                    testfile.write(datadir+"/"+subdir+"/"+file+"\t"+str(classindex)+"\n")
                fileindex+=1
            classindex=classindex+1
    datafile.close()

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Generate database needed by caffe')
    cgroup = parser.add_argument_group('Options for generating database')
    cgroup.add_argument('--root', type=str, default='../data', help='root of image to train.')
    cgroup.add_argument('--exts', type=list, default=['.jpeg', '.jpg'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0.2,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', type=bool, default=True,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--shuffle', default=True, help='If this is set as True, \
        im2rec will randomize the image order in <prefix>.lst')
    args = parser.parse_args()
#    args.root = os.path.abspath(args.root)
    return args
if __name__ == '__main__':
#    print "loading data path..."
#    generatechardatasets("data")
#    print "done"
    args = parse_args()
    make_list(args)