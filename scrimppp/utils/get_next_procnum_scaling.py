# script for computing the next number of processors in a scaling job
# the result is printed on the command line, in order to make use of the script from bash
# i just became annoyed by the bash computation capabilities. THis should be highly easier...
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("last_procnum", metavar="PREV", type=int, nargs=1, help='previous number of processors')
parser.add_argument("base", metavar='BASE', type=int, nargs=1, help='base for exponential scaling/increment for linear scaling')
parser.add_argument("--variant", choices=['linear', 'square', 'exp', 'square_exp'], default='square_exp', help='variant of scaling: linear increment by bas, next square number, exponential scaling or approximate exponential with square numbers')
args = parser.parse_args()

variant = args.variant

last_procnum=args.last_procnum[0]
base=args.base[0]
if 'square_exp' == variant:
    procnum = last_procnum
    last_exp = int(math.log(last_procnum)/math.log(base))
    # print("prev exp: %d, ideal next procnum: %d"%(last_exp, base**(last_exp+1) ))
    while procnum<=last_procnum:
        exp = last_exp+1
        ideal= base**exp
        root= int(math.sqrt(ideal))
        newnum=root*root
        procnum=newnum if (newnum<20 and newnum>last_procnum) else (root+1)*(root+1)
        print(procnum)
elif 'exp' == variant:
    last_exp = int(math.log(last_procnum) / math.log(base))
    exp = last_exp+1
    procnum = base**exp
    print(procnum)
elif 'square' == variant:
    last_root = math.sqrt(last_procnum)
    root=int(last_root)+1
    procnum=root*root
    print(procnum)
elif 'linear' == variant:
    procnum=last_procnum+base
    print(procnum)
