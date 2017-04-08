/// <reference path="jquery.d.ts" />

interface rect {
  x: number,
  y: number,
  height: number,
  width: number
}

class Blurrer {
  private image;
  private canvas;
  private ctx;
  private offset;
  private width: number;
  private height: number;
  private img_width: number;
  private img_height: number;
  private drawable: boolean;
  private aspect_ratio: number;
  private rects: Array<rect>;
  private scale: number;
  private new_rect: rect;
  private clicked: boolean = false;
  private delete: boolean = false;
  constructor(_image_url: string, old_rects: string, active: boolean) {
    this.rects = JSON.parse(old_rects);
    this.new_rect = {
      x: 0,
      y: 0,
      width: 0,
      height: 0
    };
    this.aspect_ratio = 1.0;
    this.drawable = false;

    this.canvas = $("#main-canvas");
    if (active) {
      this.canvas.on('mousemove', (e: JQueryEventObject) => {
        this.mouse_move(e.pageX - this.offset.left, e.pageY - this.offset.top);
      });
      this.canvas.click((e: JQueryEventObject) => {
        this.click(e.pageX - this.offset.left, e.pageY - this.offset.top)
      });

      $("#delete-btn").click(() => {
        this.delete = true;
      });
    }


    $(window).on('resize', () => {
      this.resize();
    });

    this.resize();


    this.ctx = this.canvas[0].getContext("2d");
    this.image = new Image();
    this.image.onload = () => {
      this.aspect_ratio = this.image.width / this.image.height;

      this.drawable = true;
      this.run();
    };
    this.image.src = _image_url;
  }


  resize() {
    this.width = 0.99*window.innerWidth ;
    this.height = this.width / this.aspect_ratio;
    this.offset = this.canvas.offset();

    this.canvas.prop("width", this.width);
    this.canvas.prop("height", this.height);

    this.img_height = this.canvas.prop("height");
    this.img_width = this.canvas.prop("width");

    if (this.drawable) {
      this.draw();
    }
  }

  draw() {
    if (!this.drawable)
      return;

    this.ctx.drawImage(this.image, 0, 0, this.img_width, this.img_height);

    if (this.clicked && this.new_rect.width > 0 && this.new_rect.height > 0) {
      this.ctx.lineWidth = 1;
      this.ctx.fillStyle = "rgba(255, 255, 0, 0.3)";
      this.ctx.fillRect(this.new_rect.x, this.new_rect.y,
        this.new_rect.width, this.new_rect.height);
    }

    this.ctx.lineWidth = 1;
    this.ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
    for (const r of this.rects) {
      this.ctx.fillRect(r.x * this.img_width, r.y * this.img_height,
                        r.width * this.img_width, r.height * this.img_height);
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
      let idx: number = -1;
      x /= this.img_width;
      y /= this.img_height;
      for (let i = 0; i < this.rects.length; ++i) {
        const r = this.rects[i];

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
            x: this.new_rect.x / this.img_width,
            y: this.new_rect.y / this.img_height,
            height: this.new_rect.height / this.img_height,
            width: this.new_rect.width / this.img_width
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