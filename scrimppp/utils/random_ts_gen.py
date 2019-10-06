import argparse
import random
import subprocess
# import numpy as np
import os

# parse command line arguments
parser = argparse.ArgumentParser(description="Generates a random walk time series of requested length with some motifs of given length (random number and multiplicities of motifs) ")
parser.add_argument("outfile", metavar="PATH", type=str, nargs=1, help='path of the outputfile')
parser.add_argument("length", metavar="N", type=int, nargs=1, help="length of the time series to produce")
parser.add_argument("motif_length", metavar="M", type=int, nargs=1, help="length of the motifs to embed")
parser.add_argument("--seed", "-s", metavar="int", type=int, nargs=1, help="optional seed for random generator")

args = parser.parse_args()

# prepare: evaluate the seed, if specified or use a default one
if args.seed and len(args.seed) > 0:
    seed = args.seed[0]
else:
    seed = 12341234
random.seed( seed)
# np.random.seed( seed )


motif_len = args.motif_length[0]
ts_len = args.length[0]
outfile = args.outfile[0]

max_embeddings = ts_len / motif_len
p = 0.2 # expected average fraction of the number of embeddings
max_embeddings = max_embeddings/2 # we also want some random walks in between of the motifs... thus at most 50% of a ts shall be covered by motifs

# randomly choose a total number of embeddings
# num_embeddings = np.random.binomial(max_embeddings, p)
rb = random.betavariate(p+1, 2-p)
num_embeddings = int( rb*(max_embeddings-1) ) +1
#print(str(rb) + " asdf" + str(num_embeddings))
# print(" num_embeddings %d / max %d"%(num_embeddings, max_embeddings))
if (num_embeddings==0):
    num_embeddings=1

# determine a number of different motifs and their multiplicity
    # the chosen approach will tend to produce motifs with high as well as low embedding nums
multiplcities = []
accu_embed = 0
while accu_embed < num_embeddings:
    k = random.randint(1, num_embeddings-accu_embed)
    multiplcities.append(k)
    accu_embed += k
num_motifs = len(multiplcities)

#print ( "num motifs: %d, total number of embeddings: %d "%(num_motifs, np.sum(multiplcities)) )
print( "chosen motif multiplicities: " + str(multiplcities) )

# invoke the more specific ts generator
gendir = os.path.dirname(os.path.realpath(__file__))
genscript = os.path.join(gendir, 'generate_ts.py')
genargs = ['python3', genscript, outfile, str(ts_len), str(num_motifs), str(motif_len)]
for e in multiplcities:
    genargs.append(str(e))
if args.seed and len(args.seed) > 0:
    genargs += ['--seed', str(args.seed[0])]
#print(genargs)
subprocess.call(genargs)