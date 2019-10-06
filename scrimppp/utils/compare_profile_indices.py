import argparse
import re, os, subprocess

# "backporting" for the sake of python 3.4.1 in lrz python 3.5 module...
# source: https://stackoverflow.com/questions/40590192/getting-an-error-attributeerror-module-object-has-no-attribute-run-while
def run(*popenargs, input=None, check=False, **kwargs):
    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = subprocess.PIPE

    process = subprocess.Popen(*popenargs, **kwargs)
    try:
        stdout, stderr = process.communicate(input)
    except:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if check and retcode:
        raise subprocess.CalledProcessError(
            retcode, process.args, output=stdout, stderr=stderr)
    return retcode, stdout, stderr

def print_verbose(msg : str):
	if (args.silent):
		return
	print(msg)

# parse the commandline arguments: get two paths to the files to compare
parser = argparse.ArgumentParser(description="Compare the profile indices of two matrix profile result files")
parser.add_argument('filename_1', metavar='PATH_1', type=str, nargs=1, help='path/filename of first matrix profile result file')
parser.add_argument('filename_2', metavar='PATH_2', type=str, nargs=1, help='path/filename of second matrix profile result file')
parser.add_argument('--keep_converted', action='store_true', help='do not remove converted ascii profiles (if conversion from bin to ascii was involved)')
parser.add_argument('--cvt_prof_path', type=str, nargs=1, help='full path of a profile conversion utility. If not specified it is assumed to be in the cwd')
parser.add_argument('--silent', action='store_true', help='suppress verbose output')
args=parser.parse_args()

#retrieve the parameters
cmp_file_1_path = args.filename_1[0]
cmp_file_2_path = args.filename_2[0]
if not args.cvt_prof_path or not len(args.cvt_prof_path) > 0:

	cvt_path = './cvt_matprof'
else:
	cvt_path = args.cvt_prof_path[0]

# convert input file 2 from bin to ascii, if necessary
converted_file_2_name = None
if re.search(".bin$", cmp_file_2_path):
	print_verbose("conversion of binary input file 2 to ASCII required...")
	#converted_file_2_name = "/tmp/temporary_matprof_2_" + cmp_file_2_path[:-5] + ".ascii"
	converted_file_2_name = "converted_" + os.path.basename(cmp_file_2_path)[:-4] + ".ascii" # remove the '.bin'
	#pyton 3.5 style:
	# cvt_result = subprocess.run([cvt_path, 'bin-to-ascii', cmp_file_2_path, '--output', converted_file_2_name], stdout=subprocess.PIPE)
	# print_verbose(cvt_result.stdout)
	# if not cvt_result.returncode == 0:
	# ...
	retcode, stdout, stderr = run([cvt_path, 'bin-to-ascii', cmp_file_2_path, '--output', converted_file_2_name], stdout=subprocess.PIPE)
	print_verbose(stdout.decode('utf-8'))
	if not retcode == 0:
		print("conversion of binary profile failed. Aborting, as only ascii profiles can be checked")
		exit(1)
	print_verbose("finished conversion to temporary file 2: " + converted_file_2_name)
	cmp_file_2_path = converted_file_2_name

# convert input file 1 from bin to ascii, if necessary
# some copy and past, just quicker. TODO: beter extract method
converted_file_1_name = None
if re.search(".bin$", cmp_file_1_path):
	print_verbose("conversion of binary input file 1 to ASCII required...")
	#converted_file_1_name = "/tmp/temporary_matprof_1_" + cmp_file_1_path[:-5] + ".ascii"
	converted_file_1_name = "converted_" + os.path.basename(cmp_file_1_path)[:-4] + ".ascii" # remove the '.bin'
	#if not 0 == subprocess.call([cvt_path, 'bin-to-ascii', cmp_file_1_path, '--output', converted_file_1_name]):
	#	print("conversion of binary profile failed. Aborting, as only ascii profiles can be checked")
	#	exit(1)
	retcode, stdout, stderr = run( [cvt_path, 'bin-to-ascii', cmp_file_1_path, '--output', converted_file_1_name], stdout=subprocess.PIPE)
	print_verbose(stdout.decode('utf-8'))
	if not retcode == 0:
		print("conversion of binary profile failed. Aborting, as only ascii profiles can be checked")
		exit(1)
	print_verbose("finished conversion to temporary file 1: " + converted_file_1_name)
	cmp_file_1_path = converted_file_1_name

difference_ctr = 0
max_profile_diff = 0
with open(cmp_file_1_path) as file1:
	with open(cmp_file_2_path) as file2:
		reached_end_of_a_file = False
		line_ctr = 0
		while not reached_end_of_a_file:
			try:
				line_ctr += 1
				left_line = file1.readline()
				right_line= file2.readline()
				#check, that none of the files has already reached its end
				if left_line or right_line:
					# obtain the matrix profile index (and distance values)
					vals_l = left_line.split()
					vals_r = right_line.split()
					if vals_l[1] != vals_r[1]:
						difference_ctr += 1
						print_verbose ("l " + str(line_ctr) + " has differing indices, id1: " + str(vals_l[1]) + " id2: " +  str(vals_r[1]))
					max_profile_diff = max(max_profile_diff, abs( float(vals_l[0])-float(vals_r[0]) ) )
				else: #the end of at least one file was reached
					reached_end_of_a_file = True
					# check, that indeed both file ends are reached, otherwise the profiles differ in length!
					if left_line or right_line:
						difference_ctr += 1
						if left_line:
							print("FATAL: file 2 is shorter than file 1!")
							while file1.readline():
								difference_ctr +=1
						else:
							print("FATAL: file 1 is shorter than file 2!")
							while file2.readline():
								difference_ctr +=1
			except BaseException as e:
				print("Exception raised while checking line %d: %s"%(line_ctr, e) )
				exit(-1)

# print a very short summary (i.e. success or failure)
if difference_ctr is 0:
	print("SUCCESS: The profile indices coincide in all " + str(line_ctr) + " entries")		
else:
	print("ERROR: Profile indices differ in " + str(difference_ctr) + " entries")
print("maximum absolute difference in profile values: " + str(max_profile_diff))

if not args.keep_converted:
	print_verbose('removing temporarily converted files')
	if converted_file_1_name:
		os.remove(converted_file_1_name)
	if converted_file_2_name:
		os.remove(converted_file_2_name)

# return the number of differences as the errorcode
exit(difference_ctr)
