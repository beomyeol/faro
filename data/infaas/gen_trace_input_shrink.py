#!/usr/bin/python3

# This script parses the twitter or alibaba traces and, given an interval and
## min/max, it will generate the traces for input into systems like INFaaS
# The script also shrinks the trace to the desired length
# Output file format is as follows:
# second num_tweets
# Note that num_tweets will be normalized between the min/max range inputted

import sys
import os
import argparse
import numpy as np
from statistics import mean

def get_args():
  ap = argparse.ArgumentParser()
  ap.add_argument('--inpfile', '-i', required=True,
                  dest='inpfile',
                  help='Input file. Must have been outputted by one of the parsing scripts')
  ap.add_argument('--min', '-m', required=True, type=int,
                  dest='min',
                  help='Minimum value to map to')
  ap.add_argument('--max', '-M', required=True, type=int,
                  dest='max',
                  help='Maximum value to map to')
  ap.add_argument('--interval', '-I', required=True, type=int,
                  dest='interval',
                  help='Interval to average over')
  ap.add_argument('--length', '-L', required=True, type=int,
                  dest='length',
                  help='Maximum length of the final trace')

  return ap.parse_args()

def main(args):
  input_file = args.inpfile
  min_map = args.min
  max_map = args.max
  interval = args.interval
  length = args.length

  if (min_map > max_map):
    print("max must be >= min")
    sys.exit(1)

  fd = open(input_file, 'r')

  raw_sec_arr = []
  raw_count_arr = []
  for line in fd:
    line_split = line.split()
    raw_sec_arr.append(int(line_split[0]))
    raw_count_arr.append(int(line_split[1]))

  fd.close()

  sec_arr = [] 
  count_arr = [] 
  if length < len(raw_sec_arr):
    # Map trace to desired length
    new_range = np.arange(0, len(raw_sec_arr), len(raw_sec_arr)/length)

    for x in range(len(new_range)-1):
      sec_arr.append(x)
      start_ind = int(new_range[x])
      end_ind = int(new_range[x+1])
      # Include last value
      if x == len(new_range)-1:
        end_ind += 1
      count_arr.append(int(mean(raw_count_arr[start_ind:end_ind])))
  else:
    sec_arr.extend(raw_sec_arr)
    count_arr.extend(raw_count_arr)

  new_sec_arr = []
  new_count_arr = []
  # Average over interval
  interval_start = 0
  interval_end = 1
  running_sec = 0
  while interval_end < len(sec_arr):
    # Find the end of the interval
    while (sec_arr[interval_end] - sec_arr[interval_start]) < interval:
      interval_end += 1
      if interval_end == len(sec_arr):
        break

    # Average and save
    next_mean = int(mean(count_arr[interval_start:interval_end]))

    new_sec_arr.append(running_sec)
    new_count_arr.append(next_mean)

    running_sec += interval
    interval_start = interval_end
    interval_end += 1

  out_name = os.path.basename(input_file)
  out_name = out_name.split('.')[0]
  out_name = out_name + '_norm.txt'

  fd_wr = open(out_name, 'w')

  # Find the minimum
  min_trace = min(new_count_arr)

  # Find the maximum
  max_trace = max(new_count_arr)

  # Set up ranges for mapping
  input_range = max_trace - min_trace
  output_range = max_map - min_map

  # Overwrite to new range
  for i,x in enumerate(new_count_arr):
    next_mapping = (x - min_trace)*output_range / input_range + min_map
    fd_wr.write('%d %d\n' % (new_sec_arr[i], next_mapping))

  fd_wr.close()

  print('Done!!')

if __name__ == '__main__':
  main(get_args())

