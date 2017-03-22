from flask import Flask, url_for, redirect, request, render_template, send_from_directory, jsonify
from glob import glob
from os.path import isfile
from fastnumbers import fast_real

app = Flask(__name__, static_url_path='', template_folder='templates')

@app.route("/")
def index():
  return redirect(url_for('static', filename='index.html'))

@app.route('/api/get-prev', methods=['POST'])
def get_prev():
  data = request.get_json()
  floor_id = data['floor_id']
  img_name = data['img_name']

  image_urls = glob("images/{}/*.png".format(floor_id))
  image_urls.sort();

  idx = image_urls.index("images/{}/{}".format(floor_id, img_name));
  if idx > 0:
    return jsonify(image_urls[idx - 1])
  else:
    return jsonify(image_urls[idx])

@app.route('/api/get-next', methods=['POST'])
def get_next():
  data = request.get_json()
  floor_id = data['floor_id']
  img_name = data['img_name']

  image_urls = glob("images/{}/*.png".format(floor_id))
  image_urls.sort();

  idx = image_urls.index("images/{}/{}".format(floor_id, img_name));

  if idx < len(image_urls) - 1:
    return jsonify(image_urls[idx + 1])
  else:
    return jsonify(image_urls[idx])

@app.route('/api/save', methods=['POST'])
def save():
  data = request.get_json()
  rects = data['rects']
  img_data = data['img_data']
  floor_id = img_data['floor_id']
  img_name = img_data['img_name']

  with open("rects/{}/{}".format(floor_id, img_name.replace(".png", ".txt")), "w+") as f:
    for r in rects:
      f.write("{} {} {} {}\n".format(r['x'], r['y'], r['width'], r['height']))

  return jsonify(True)

@app.route('/api/load-old', methods=['POST'])
def load_old_rects():
  data = request.get_json()
  floor_id = data['floor_id']
  img_name = data['img_name']
  prev_file = "rects/{}/{}".format(floor_id, img_name.replace(".png", ".txt"))
  prev_rects = []
  if isfile(prev_file):
    with open(prev_file, "r") as f:
      data = f.read().strip().split("\n")
      for l in data:
        rect = l.split(" ")
        x, y, width, height = rect[0], rect[1], rect[2], rect[3]
        prev_rects.append({
                            "x": fast_real(x),
                            "y": fast_real(y),
                            "width": fast_real(width),
                            "height": fast_real(height)
                          })
  return jsonify(prev_rects)


@app.route('/floor/<floor_id>')
def floor(floor_id):
  image_urls = glob("images/{}/*.png".format(floor_id))
  image_urls.sort();

  images = []
  for url in image_urls:
    images.append({
                    "img_name": url.split("/")[2],
                    "url": url
                  })

  return render_template('floor.html', floor_id=floor_id, images=images, blurable=True)

@app.route('/images/<path:img>')
def floor_path(img):
  return send_from_directory('images', img)


@app.route('/blur/<path:img>')
def blur(img):
  return render_template('blur.html', img=img, floor_id=img.split("/")[1])

@app.route('/js/<floor_id>')
def send_js(floor_id):
  return send_from_directory('scripts/js', floor_id)



if __name__ == '__main__':
  app.run()