import serial
import argparse
from datetime import datetime
from time import sleep

fn = "power_log.csv"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--output", type=str, required=False)
	FLAGS = parser.parse_args()

	if FLAGS.output:
		fn = FLAGS.output

	mcu = serial.Serial(port="/dev/ttyUSB0", baudrate=1000000, timeout=1)
	while True:
		data = mcu.readline().decode('utf-8')#.rstrip()
		if data:
			f = open(fn, 'a')
			f.write(datetime.now().strftime('%H:%M:%S.%f')[:-3])
			f.write(",")
			f.write(data)
			f.close()
