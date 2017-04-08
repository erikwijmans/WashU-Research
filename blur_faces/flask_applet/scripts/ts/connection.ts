/// <reference path="jquery.d.ts" />


function ajax_POST(url: string, data: string, cb) {
  $.ajax({
    url: url,
    dataType: "json",
    type: "POST",
    contentType: 'application/json;charset=UTF-8',
    data: data,
    success: cb,
    error: (err) => {
      console.log(err);
    }
  });
}

function ajax_GET(url: string, data, cb) {
  $.ajax({
    url: url,
    dataType: "json",
    type: "GET",
    contentType: 'application/json;charset=UTF-8',
    data: data,
    success: cb,
    error: (err) => {
      console.log(err);
    }
  });
}

function set_up_connections(floor_id: string, img_name: string,
  save_url: string,
  blurrer: Blurrer) {

  let img_data = {
    floor_id: floor_id,
    img_name: img_name
  };
  let prev_btn = $("#prev-btn");
  let next_btn = $("#next-btn");
  let save_btn = $("#save-btn");

  let save_fn = () => {
    let data: string = JSON.stringify({
                                          rects: blurrer.get_rects(),
                                          img_data: img_data
                                      });
    ajax_POST(save_url, data, (data) => {
      if (data) {
        save_btn.tooltip('show');
        setTimeout(() => {
          save_btn.tooltip('hide');
        }, 2000);
      }
    })
  };

  prev_btn.click(save_fn);
  next_btn.click(save_fn);
  save_btn.click(save_fn);

}