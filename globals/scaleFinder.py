#!/usr/bin/env python

# Usage:  ./scale_finder.py <dataPath> <start_scale> <number> <extras>
# extras are flags that will be passed to both placeScan and scanDensity

from sys import argv
import subprocess, os

place_exe = './placeScan/placeScan'
density_exe = './scanDensity/scanDensity'

def inc_scale(current_scores, current_scales, scale, inc):
  current_scores = current_scores[1:]
  current_scores.append(0)
  print current_scores
  current_scales = current_scales[1:]
  current_scales.append(scale + inc)
  print current_scales
  return current_scores, current_scales

def sub_scale(current_scores, current_scales, scale, inc):
  current_scores = current_scores[:2]
  current_scores.insert(0, 0)

  current_scales = current_scales[:2]
  current_scales.insert(0, scale + inc)

  return current_scores, current_scales

def change_delta(current_scores, current_scales, scale, inc):
  current_scores = current_scores[1:2]
  current_scores.append(0)
  current_scores.insert(0, 0)

  current_scales = current_scales[1:2]
  current_scales.append(scale + inc)
  current_scales.insert(0, scale - inc)

  return current_scores, current_scales

def main():
  if len(argv) < 4:
    print 'Not the correct number of arguements!'
    return

  dataPath = argv[1]
  start_scale = float(argv[2])
  number = int(argv[3])
  extras = ' '.join(argv[4:])
  output_dir = '{}/placementOptions/V1'.format(dataPath)
  scale_file_name = '{}/scale.txt'.format(dataPath)

  scale_delta = 0.5
  result_name = ''
  current_scores = [0, 0, 0]
  current_scales = []
  for i in range(-1, 2):
    current_scales.append(start_scale + i*scale_delta)

  while scale_delta > 0.0001:
    print scale_delta
    for i in range(0, len(current_scores)):
      if current_scores[i] != 0:
        continue

      scale = current_scales[i]

      command = '{} -dataPath={} -redo -2D -numScans=1 -startNumber={} -scale={} {}'.format(density_exe, dataPath, number, scale, extras)
      print '\n{}'.format(command)
      subprocess.call(command.split(' '))

      command = '{} -dataPath={} -redo -V1 -numScans=1 -startNumber={} -noerrosion {}'.format(place_exe, dataPath, number, extras)
      print '\n{}'.format(command)
      subprocess.call(command.split(' '))

      if len(result_name) == 0:
        for f in os.listdir(output_dir):
          if f.find('{:03d}.txt'.format(number)) != -1:
            result_name = f
            break

      results_file = open('{}/{}'.format(output_dir, result_name), 'r')
      #Trash the header line
      results_file.readline()
      #The best score is the first element in a space seperated list
      count = 0
      average = 0
      for line in iter(lambda: results_file.readline(), ''):
        average += float(line.split(' ')[0])
        if ++count == 20:
          break
      results_file.close()
      current_scores[i] = average/count


    print '\nscores: {}'.format(current_scores)
    if current_scores[0] < current_scores[1] and current_scores[2] < current_scores[1]:
      if current_scores[0] < current_scores[2]:
        current_scores, current_scales = sub_scale(current_scores, current_scales, current_scales[0], scale_delta)
      else:
        current_scores, current_scales = inc_scale(current_scores, current_scales, current_scales[2], scale_delta)
    elif current_scores[0] < current_scores[1]:
      current_scores, current_scales = sub_scale(current_scores, current_scales, current_scales[0], scale_delta)
    elif current_scores[2] < current_scores[1]:
      current_scores, current_scales = inc_scale(current_scores, current_scales, current_scales[2], scale_delta)
    else:
      scale_delta -= 0.1
      current_scores, current_scales = change_delta(current_scores, current_scales, current_scales[1], scale_delta)

  scale_file = open(scale_file_name, 'w')
  scale_file.write('{}'.format(current_scales[1]))
  scale_file.close()

if __name__ == '__main__':
  main()