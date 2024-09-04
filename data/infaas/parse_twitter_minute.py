#!/usr/bin/python3

# This script parses the twitter traces and outputs the number of tweets PER MINUTE
# Thus, it will do this for the specified day directory
# Output file format is as follows:
# minute num_tweets

import sys
import os
import bz2
import glob
import json
import argparse

def get_args():
  ap = argparse.ArgumentParser()
  ap.add_argument('--daydir', '-d', required=True,
                  dest='daydir',
                  help='Full path to day directory (e.g., ../02/06/)')

  return ap.parse_args()

def parse(input_dir):
  output_prefix = input_dir.split('/')
  if output_prefix[-1] == '':
    output_prefix = output_prefix[:-1]
  output_prefix = '_'.join(output_prefix[-2:])

  # Create files by going from 00 to 59. This is to process them
  ## in chronological order. Skip if it doesn't exist

  min_dict = {} # K: time, V: counter

  for hour in range(24):
    for minute in range(60):
      next_file = os.path.join(
          input_dir, "{:02d}".format(hour), "{:02d}".format(minute)) + '.json.bz2'

      if not os.path.exists(next_file):
        print('Skipping %s; does not exist' % next_file)
        continue

      print ('Processing:', next_file)

      fd = bz2.open(next_file, 'rt')
      counter = 0
      for line in fd:
        one_tweet = json.loads(line)
        if 'created_at' in one_tweet:
          counter += 1 
      fd.close()
      general_min = hour*60 + minute
      min_dict[general_min] = counter

  fd_out = open('twitter_' + output_prefix + '.txt', 'w')
  for k,v in min_dict.items():
    fd_out.write('%d %d\n' % (k,v))

  fd_out.close()

  print('Done!!')

if __name__ == '__main__':
  args = get_args()
  input_dir = args.daydir
  parse(input_dir)

