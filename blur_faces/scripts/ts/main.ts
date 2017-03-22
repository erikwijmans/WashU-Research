/// <reference path="jquery.d.ts" />

interface rect {
  x: number,
  y: number,
  height: number,
  width: number
}

class Blurrer {
  private image_url: string;
  private image;
  private canvas;
  private ctx;
  private offset;
  private width: number;
  private height: number;
  private drawable: boolean;
  private aspect_ratio: number;
  private rects: Array<rect>;
  private scale: number;
  private new_rect: rect;
  private clicked: boolean = false;
  private delete: boolean = false;
  constructor(_image_url: string, old_rects: Array<rect>) {
    this.rects = old_rects;
    this.new_rect = {
      x: 0,
      y: 0,
      width: 0,
      height: 0
    };
    this.aspect_ratio = 1.0;
    this.drawable = false;
    this.image_url = _image_url;
    this.canvas = $("#main-canvas");
    this.canvas.on('mousemove', (e: JQueryEventObject) => {
      this.mouse_move(e.pageX - this.offset.left, e.pageY - this.offset.top);
    });
    this.canvas.click((e: JQueryEventObject) => {
      this.click(e.pageX - this.offset.left, e.pageY - this.offset.top)
    });

    $("#delete-btn").click(() => {
      this.delete = true;
    });


    $(window).on('resize', () => {
      this.resize();
    });

    this.resize();



    this.ctx = this.canvas[0].getContext("2d");
    this.image = new Image();
    this.image.onload = () => {
      this.aspect_ratio = this.image.width / this.image.height;
      console.log(this.aspect_ratio);
      this.drawable = true;
      this.run();
    };
    this.image.src = `/${this.image_url}`;
  }


  resize() {
    this.width = window.innerWidth;
    this.height = this.width / this.aspect_ratio;
    this.offset = this.canvas.offset();


    this.canvas.prop("width", this.width);
    this.canvas.prop("height", this.height);

    if (this.drawable) {
      this.draw();
    }
  }

  draw() {
    if (!this.drawable)
      return;

    this.ctx.drawImage(this.image, 0, 0, this.width, this.height);

    if (this.clicked && this.new_rect.width > 0 && this.new_rect.height > 0) {
      this.ctx.lineWidth = 1;
      this.ctx.fillStyle = "rgba(255, 255, 0, 0.3)";
      this.ctx.fillRect(this.new_rect.x, this.new_rect.y,
        this.new_rect.width, this.new_rect.height);
    }

    this.ctx.lineWidth = 1;
    this.ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
    for (let r of this.rects) {
      this.ctx.fillRect(r.x * this.width, r.y * this.height, r.width * this.width, r.height * this.height);
    }
  }

  get_rects() {
    return this.rects;
  }

  mouse_move(x: number, y: number) {
    this.new_rect.width = Math.max(0, x - this.new_rect.x);
    this.new_rect.height = Math.max(0, y - this.new_rect.y);
  }

  click(x: number, y: number) {
    if (this.delete) {
      var idx: number = -1;
      x /= this.width;
      y /= this.height;
      for (var i = 0; i < this.rects.length; ++i) {
        let r = this.rects[i];

        if (x > r.x && y > r.y && x < (r.x + r.width) && y < (r.y + r.height)) {
          idx = i;
        }
      }
      if (idx > -1) {
        console.log(`Delete ${idx}`);
        this.rects.splice(idx, 1);
      }
      this.delete = false;
    } else {

      if (!this.clicked) {
        this.new_rect.x = x;
        this.new_rect.y = y;
      } else {
        if (this.new_rect.width > 0 && this.new_rect.height > 0) {
          this.rects.push({
            x: this.new_rect.x / this.width,
            y: this.new_rect.y / this.height,
            height: this.new_rect.height / this.height,
            width: this.new_rect.width / this.width
          });
        }
        this.new_rect.width = 0;
        this.new_rect.height = 0;
      }


      this.clicked = !this.clicked;
    }
  }


  run() {
    this.resize();

    window.requestAnimationFrame(() => {
      this.run();
    });
  }
}