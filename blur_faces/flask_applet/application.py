from flask import Flask, url_for, redirect, request, render_template, send_from_directory, jsonify, current_app
from glob import glob
from os.path import isfile
from fastnumbers import fast_real
import re, sys

app = Flask(__name__)


searcher1 = re.compile(r"""
                      .*? #Matches everything upto the floor name
                      ([A-Za-z]{3}) #Matches the floor name
                      _.*?_ #Matches _<anything>_
                      (\d{3}) #Matches the building name
                      \.[a-z]{3} #Matches the type
                      """, re.VERBOSE)
searcher2 = re.compile(r"""
                      .*? #Matches everything upto the floor name
                      ([A-Za-z]{3}) #Matches the floor name
                      (\d{3}) #Matches the building name
                      \.[a-z]{3} #Matches the type
                      """, re.VERBOSE)
def parse_name(name):
  res = searcher1.search(name)
  if res is None:
    res = searcher2.search(name)
  return res.group(1), res.group(2)

@app.route('/api/save', methods=['POST'])
def save():
  data = request.get_json()
  rects = data['rects']
  img_data = data['img_data']
  floor_id = img_data['floor_id']
  img_name = img_data['img_name']
  building, num = parse_name(img_name)

  with open("rects/{}/{}_rects_{}.txt".format(floor_id, building, num), "w+") as f:
    for r in rects:
      f.write("{} {} {} {}\n".format(r['x'], r['y'], r['width'], r['height']))

  return jsonify(True)


@app.route('/floor', methods=['GET'])
def floor():
  floor_id = request.args.get('floor_id', "DUC1")
  active = request.args.get('active', False)
  image_urls = glob("ReleaseData/{}/imgs/*.png".format(floor_id))
  image_urls.sort()

  images = []
  for url in image_urls:
    images.append({
                    "img_name": url.split("/")[-1],
                    "floor_id": floor_id
                  })

  return render_template('floor.html', floor_id=floor_id, images=images, active=active)


def load_old_rects(floor_id, img_name):
  building, num = parse_name(img_name)
  prev_file = "rects/{}/{}_rects_{}.txt".format(floor_id, building, num)
  prev_rects = []
  if isfile(prev_file):
    with open(prev_file, "r") as f:
      data = [l.strip() for l in f.read().split("\n") if len(l.strip()) > 0]
      for l in data:
        rect = l.split(" ")
        x, y, width, height = rect[0], rect[1], rect[2], rect[3]
        prev_rects.append({
                            "x": fast_real(x),
                            "y": fast_real(y),
                            "width": fast_real(width),
                            "height": fast_real(height)
                          })
  return prev_rects


@app.route('/annotate', methods=['GET'])
def annotate():
  floor_id = request.args.get('floor_id', "DUC1")
  img_name = request.args.get('img_name', "DUC_pano_000.png")
  active = request.args.get('active', False)

  image_urls = glob("ReleaseData/{}/imgs/*.png".format(floor_id))
  image_urls.sort();

  idx = image_urls.index("ReleaseData/{}/imgs/{}".format(floor_id, img_name));

  if idx < len(image_urls) - 1:
    next_url = image_urls[idx + 1]
  else:
    next_url = image_urls[idx]

  if idx > 0:
    prev_url = image_urls[idx - 1]
  else:
    prev_url = image_urls[idx]

  prev_name = prev_url.split("/")[3]
  next_name = next_url.split("/")[3]

  return render_template('annotate.html', img_name=img_name, floor_id=floor_id, active=active, prev_rects=load_old_rects(floor_id, img_name), prev_name=prev_name, next_name=next_name)

@app.route("/", methods=['GET'])
def main():
  active = request.args.get('active', False)
  return render_template('main.html', active=active)

@app.route('/js/<script_name>')
def send_js(script_name):
  return send_from_directory('scripts/js', script_name)

@app.route('/api/images', methods=['GET'])
def get_image():
  floor_id = request.args.get('floor_id', "DUC1")
  img_name = request.args.get('img_name', "DUC_pano_000.png")
  lowres = request.args.get('lowres', False)
  print(floor_id, img_name, lowres)
  if not lowres:
    file = "{}/imgs/{}".format(floor_id, img_name)
  else:
    file = "{}/lowres/{}".format(floor_id, img_name)

  return send_from_directory("ReleaseData", file)


if __name__ == '__main__':
  app.run()