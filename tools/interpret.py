import numpy as np
import os
import pandas as pd
import argparse


'''
Input: the filename of the scheduler log file
Output: average energy, average released, average missed
'''
def sched_interpreter(fn):
	df_sched = pd.read_csv(fn, header=None)
	df_sched.columns = ["filename", "released", "missed", "energy"]

	l1 = df_sched["energy"]
	l2 = df_sched["released"]
	l3 = df_sched["missed"]

	return sum(l1) / len(l1), sum(l2) / len(l2), sum(l3) / len(l3)

# convert the format H:M:S.MS to ms
# input as a string
def convert_datetime_to_millis(time_string):
	h, m, s = time_string.split(':')
	s, ms = s.split('.')
	return (int(h) * 3600 * 1000 + int(m) * 60 * 1000 + int(s) * 1000 + int(ms))

'''
Input: list of timestamps from power log
Output: the index
'''
def find_neareast_timestamp(tp_list, target, checkpoint):
	tg_ms = convert_datetime_to_millis(target)
	for i in range(checkpoint, len(tp_list)):
		if (type(tp_list[i]) == float):
			print("{}, {}".format(i, tp_list[i]))
		try:
			a, b, c = tp_list[i].split(':')
		except ValueError:
			print("Not a valid time {} at {}".format(tp_list[i], i))
		tp = convert_datetime_to_millis(tp_list[i])
		if tp == tg_ms:
			return i
		elif tp > tg_ms:
			return i - 1

# These lists stores the results of all the groups of tasksets
pred_list = []
released_list = []
missed_list = []
ratio_list = []
energy_list = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--start", type=int, required=True)
	parser.add_argument("-e", "--end", type=int, required=True)
	parser.add_argument("-m", "--mode", type=str, required=True)
	FLAGS = parser.parse_args()

	start = FLAGS.start
	end = FLAGS.end
	mode = FLAGS.mode

	output_folder = "../output/taskset_08022022/" + mode
	step = 2

	for u in range(start, end + 2, step):
		su = str(u).zfill(2)
		fn_s = os.path.join(output_folder, "set_u" + su + ".csv")
		# fn_p = os.path.join(output_folder, "power_u" + su + ".csv")
		no = str(start).zfill(2) + str(end).zfill(2)
		fn_p = os.path.join(output_folder, "power_" + no + ".csv")
		e_pred, r, m = sched_interpreter(fn_s)
		pred_list.append(e_pred)
		released_list.append(r)
		missed_list.append(m)
		ratio = float(m) * 100 / float(r)
		ratio_list.append(ratio)

		print("{},{}".format(e_pred, ratio))
	