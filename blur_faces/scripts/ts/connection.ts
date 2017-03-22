/// <reference path="jquery.d.ts" />


function ajax_req(url: string, data: string, cb) {
  $.ajax({
    url: url,
    dataType: "json",
    type: "POST",
    contentType: 'application/json;charset=UTF-8',
    data: data,
    success: (data) => {
      cb(data);
    },
    error: (err) => {
      console.log(err);
    }
  });
}

function set_up_connections(img_url: string, get_prev_url: string,
  save_url: string, get_next_url: string,
  blurrer: Blurrer) {
  let img_data = {
    floor_id: img_url.split("/")[1],
    img_name: img_url.split("/")[2]
  };
  let prev_btn = $("#prev-btn");
  let next_btn = $("#next-btn");
  let save_btn = $("#save-btn");

  let save_fn = () => {
    let data: string = JSON.stringify({ rects: blurrer.get_rects(), img_data: img_data });
    ajax_req(save_url, data, (data) => {
      if (data) {
        save_btn.tooltip('show');
        setTimeout(() => {
          save_btn.tooltip('hide');
        }, 2000);
      }
    })
  };

  prev_btn.click(() => {
    save_fn();
    ajax_req(get_prev_url, JSON.stringify(img_data), (data) => {
      window.location.replace(`/blur/${data}`);
    });
  });

  next_btn.click(() => {
    save_fn();
    ajax_req(get_next_url, JSON.stringify(img_data), (data) => {
      window.location.replace(`/blur/${data}`);
    });
  });

  save_btn.click(save_fn);
}