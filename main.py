from src.scheduler import Run
import argparse
import os, sys
sys.path.append("src/model")
sys.path.append("src/algorithm")


workspace_folder = ""
filepath = workspace_folder + "example/taskset_08022022/"

def run(policy: str, util: float, duration: float, checkpoint: int, verbose: bool, emode: bool):
	print("Running multiple tasksets from folder...")
	file_list = []
	suffix = str(util).zfill(2)
	folder = "set_u" + suffix

	for i in range(checkpoint, 50):
		fn = os.path.join(filepath, folder, str(i).zfill(3) + ".csv")
		print(fn)
		Run(fn, duration, policy, verbose, emode)

def run_single(policy: str, duration: float, filename: str, verbose: bool, emode: bool):
	print("Running a single taskset...")
	Run(filename, duration, policy, verbose, emode)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--duration", type=float, required=True)
	parser.add_argument("-p", "--policy", type=str, required=True)
	parser.add_argument("-c", "--checkpoint", type=int, required=False)
	parser.add_argument("-f", "--filename", type=str, required=False)
	parser.add_argument("-u", "--util", type=int, required=False) # The util of the target taskset
	parser.add_argument("-m", "--emode", type=bool, required=False)
	FLAGS = parser.parse_args()

	checkpoint = 0
	if FLAGS.checkpoint:
		checkpoint = FLAGS.checkpoint
	
	emode = 0
	if FLAGS.emode:
		emode = FLAGS.emode

	if FLAGS.filename: # run a single taskset
		run_single(FLAGS.policy, FLAGS.duration * 1000, FLAGS.filename, True, emode)
	elif FLAGS.util:
		run(FLAGS.policy, FLAGS.util, FLAGS.duration * 1000, checkpoint, False, emode)
	else:
		print("Error. At least on of --filename and --util must be specified!")
		exit(1)

