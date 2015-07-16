#uses ffmpeg to create a series of stills from a video
#fist arg is path to the videofile and filename
#second arg is the path to the output folder
#third arg is the FPS



import subprocess, sys, os

args = sys.argv

if len(args) != 4 and len(args) !=3 :
	print("Usage : python videoToImages.py <input_video> <output_folder>")
	print("OR Usage :python videoToImages.py <input_video> <output_folder> <frame_rate>")
	sys.exit();

output = args[2] + 'image-%5d.png'
if len(args) == 4 :
	subprocess.call(['ffmpeg', '-i', args[1], '-r', args[3], output])

else:
	subprocess.call(['ffmpeg', '-i', args[1], output])