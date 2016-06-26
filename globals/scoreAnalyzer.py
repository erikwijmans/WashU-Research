#!/usr/bin/env python

from sys import argv
import os, math

def analyze(data):
  average = 0.0
  for s in data:
    average += s

  average /= len(data)

  sigma = 0.0
  for s in data:
    sigma += (s - average)*(s - average)
  sigma /= len(data) - 1
  sigma = math.sqrt(sigma)

  print 'min: {}, max: {}, average: {},  sigma: {}'.format(min(data), max(data), average, sigma)

def main():
  data_path = argv[1]
  results_file = '{}/placementOptions/V1'.format(data_path)
  scores = []
  deltas = []
  totalDeltas = []
  for f in os.listdir(results_file):

    if f.find('.txt') != -1:
      file = open('{}/{}'.format(results_file, f), 'r')
      file.readline()
      firstScore = float(file.readline().split(' ')[0])
      scores.append(firstScore)
      for line in iter(lambda: file.readline(), ''):
        score = float(line.split(' ')[0])
        delta = score - scores[-1]
        totalDelta = score - firstScore
        totalDeltas.append(totalDelta)
        scores.append(score)
        if delta > 0.0001:
          deltas.append(delta)

  analyze(scores)
  analyze(deltas)
  analyze(totalDeltas)
  print '\n'
if __name__ == '__main__':
  main()