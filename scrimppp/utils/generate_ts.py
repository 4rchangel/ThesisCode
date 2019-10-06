import argparse
import random
#import numpy as np

# random sampling without replacement from a huge range, avoiding to enumerate it:
# from Python Cookbook, 2nd Edition, By David Ascher, Alex Martelli, Anna Ravenscroft, Publisher O'Reilly Media
# https://resources.oreilly.com/examples/9780596007973/blob/master/cb2_examples/cb2_18_4_sol_1.py
# snippet usage permitted by O'Reilly Media https://resources.oreilly.com/examples/9780596007973/blob/master/README.md
def sample(n, r):
    " Generate r randomly chosen, sorted integers from [0,n) "
    rand = random.random
    pop = n
    for samp in range(r, 0, -1):
        cumprob = 1.0
        x = rand( )
        while x < cumprob:
            cumprob -= cumprob * samp / pop
            pop -= 1
        yield n-pop-1

# parse command line arguments
parser = argparse.ArgumentParser(description="Generates a random walk time series with some embedded random motifs with the specified length and numbers of occurences")
parser.add_argument("outfile", metavar="PATH", type=str, nargs=1, help='path of the outputfile')
parser.add_argument("length", metavar="N", type=int, nargs=1, help="length of the time series to produce")
parser.add_argument("motif_count", metavar="K", type=int, nargs=1, help="number of motifs to embedd")
parser.add_argument("motif_length", metavar="M", type=int, nargs=1, help="length of the motifs to embed")
parser.add_argument("embedding_nums", metavar="E", type=int, nargs="+", help="number of times a specific motif shall appear in the ouput. Exactly K appearance numbers need to be specified")
parser.add_argument("--seed", "-s", metavar="int", type=int, nargs=1, help="optional seed for random generator")
parser.add_argument("--random_walk_amplitude", type=float, default=100,  help="optional amplitude for random walk (scaling factor for random numbers")
parser.add_argument("--motif_amplitude", type=float, default=100, nargs=1, help="optional amplitude for motifs (scaling factor fo random numbers)")
parser.add_argument("--kind", choices=['randwalk', 'noise'], default='randwalk', help="random walk or noise series?")
# TODO: add additional parameters, e.g. amplitudes, noise levels, random seed or the possibilit to specify a distinct motif, eg. by loading from a file
args = parser.parse_args()

# arbitrarly set amplitude:
motif_amplitude = args.motif_amplitude
noise_amplitude = 10
motif_rescale_min = 0.5
motif_rescale_max = 1.5
rand_walk_amplitude = args.random_walk_amplitude

# kind of random ts
if args.kind == 'randwalk':
	print("producing random walk")
	randwalk=True
elif args.kind == 'noise':
	print("producing noise time series")
	randwalk=False
else:
	print("TS kind not set, producing random walk")
	randwalk=True

# validate the arguments, i.e. the E multiplicity and admissibility of the lengths
motif_count = args.motif_count[0]
ts_len = args.length[0]
if not motif_count == len(args.embedding_nums):
	print("ERROR: The number of specified motif embeddigns and the desired motif_count need to match! Asked for {} motifs but specified {} embedding_nums".format(motif_count, len(args.embedding_nums)))
total_motif_count = sum(args.embedding_nums)
total_motif_len = total_motif_count * args.motif_length[0]
if total_motif_len > ts_len:
	print("ERROR: The accumulated length of all motif embeddings exceeds the total time series lenght. Please adjust your configuration!")

# prepare: evaluate the seed, if specified or use a default one
if args.seed and len(args.seed) > 0:
    seed = args.seed[0]
else:
    seed = 12341234
random.seed( seed)
#np.random.seed( seed )

motif_deltas = []
motif_ids = []
# prepare: generate random motifs with the desired length
for i_mot in range(motif_count):
	motif_length = args.motif_length[0]
	motif_deltas.append([ (random.random()-0.5)*motif_amplitude for i in range(motif_length)] )
	for i in range(args.embedding_nums[i_mot]):
		motif_ids.append(i_mot)

# create a random ordering for the motifs to appear
random.shuffle(motif_ids)

# create random motif positions within a series reduce by the total motif length
meta_length = ts_len-total_motif_len+total_motif_count # time series length, if each motif had length 1. From that we randomly will choose the motif positions, in order to insert them at the repsctive positions
get_next_motif_position = sample( meta_length, total_motif_count)

# generate the time series
# TODO. store the motifs and their positions into some file for proper testing/verification
motif_meta = zip(motif_ids, get_next_motif_position)
next_motif_id, next_motif_pos = next(motif_meta)
val = (random.random()-0.5)*rand_walk_amplitude	# initial value in the time series
pos = 0
with open(args.outfile[0], "w") as of:
	for i in range(meta_length):
		if i == next_motif_pos:
			print("motif " + str(next_motif_id) + " @ " + str(pos))
			# append the motif to the ouput with some random rescaling
			scale = motif_rescale_min + random.random()*(motif_rescale_max-motif_rescale_min)
			for delta in motif_deltas[next_motif_id]:
				tmp = delta*scale + noise_amplitude*(random.random()-0.5)
				if randwalk:
					val+=tmp
				else:
					val=tmp
				of.write(str(val) + " ")
				pos +=1
			next_motif_id, next_motif_pos = next(motif_meta, (None, None))
		else:
			delta = (random.random()-0.5)*rand_walk_amplitude
			if randwalk:
				val += delta
			else:
				val = delta
			of.write(str(val) + " ")
			pos +=1
print("Generated the time series!")
