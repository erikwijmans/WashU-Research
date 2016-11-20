#!/usr/bin/env python3

import sys, subprocess, shlex, glob

exe_name = sys.argv[1]
data_dir = sys.argv[2]
exe_dir = "/".join(exe_name.split("/")[:-1])

command = "make --no-print-directory -j4 -C {}".format(exe_dir)
print(command)
subprocess.check_call(shlex.split(command))

for name in glob.glob("{}/*".format(data_dir)):
  if name.find(".ply"):
    cloud_name = name.split("/")[-1]
    path = "/".join(name.split("/")[:-1])
    command = "{} -dataPath {} -outputV2 ./ -cloud_name {} -omega 0.1 -d_velocity 6.0 -h_velocity 4.0 -FPS 60".format(exe_name, path, cloud_name)
    print(command)
    subprocess.check_call(shlex.split(command))

    command = "ffmpeg ffmpeg -framerate 60 -f image2 -s 1920x1080 -i {}/record/img%06d.png -c:v libx264 -preset veryslow -threads 4 -crf 30 -r 60 -pix_fmt yuv420p {}/out.mp4".format(exe_dir, exe_dir)
    print(command)
    subprocess.check_call(shlex.split(command))