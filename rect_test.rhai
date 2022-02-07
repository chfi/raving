

export let out_image = image_var("out_image");
export let out_view = image_view_var("out_view");


let bg_bindings = [vk::binding::storage_image(0)];
export let bg_pipeline = load_compute_shader("shaders/trig_color.comp.spv",
                                      bg_bindings,
                                      8);


let bg_inputs = [#{ type: "image_view", binding: 0, view: out_view }];

export let bg_desc_set = create_desc_set(vk::ShaderStageFlags::COMPUTE,
                                         bg_bindings,
                                         bg_inputs);


fn init() {
   let init_batch = batch_builder();
   init_batch
}

fn background(w, h, t) {
  let batch = batch_builder();
  let img = self::out_image.get();

  let pc = blob();
  pc.append_int(w);
  pc.append_int(h);

  let x_size = 16;
  let y_size = 16;

  let x_groups = (w / x_size) + w % x_size;
  let y_groups = (h / y_size) + h % y_size;

  batch.transition_image(img,
                         vk::AccessFlags::empty(),
                         vk::PipelineStageFlags::TOP_OF_PIPE,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::UNDEFINED,
                         vk::ImageLayout::GENERAL);

  batch.dispatch_compute(self::bg_pipeline.get(),
                         self::bg_desc_set.get(),
                         pc, x_groups, y_groups, 1);

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::TRANSFER_READ,
                         vk::PipelineStageFlags::TRANSFER,
                         vk::ImageLayout::GENERAL,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

  batch
}

/*
fn fill_rect(batch, x, y, w, h) {
  let img = self::out_image.get();

  let pc = blob();
  pc.append_float(1.0);
  pc.append_float(0.0);
  pc.append_float(0.0);
  pc.append_float(1.0);
  pc.append_int(x);
  pc.append_int(y);
  pc.append_int(w);
  pc.append_int(h);

  let x_size = 16;
  let y_size = 16;

  let x_groups = (w / x_size) + w % x_size;
  let y_groups = (h / y_size) + h % y_size;

  batch.dispatch_compute(self::fill_pipeline.get(),
                         self::bg_desc_set.get(),
                         pc, x_groups, y_groups, 1);

  batch
}
*/


fn foreground(w, h, t) {
  let batch = batch_builder();
  let img = self::out_image.get();
  let line_count = self::line_count.get();

  let x = 400.0 + 200.0 * t.sin();
  let y = 300.0 + 160.0 * t.cos();

  let pc = blob();
  pc.append_int(x.to_int());
  pc.append_int(y.to_int());
  pc.append_int(w);
  pc.append_int(h);

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                         vk::ImageLayout::GENERAL);

  batch.dispatch_compute(self::line_pipeline.get(),
                         self::line_desc_set.get(),
                         pc, 80, line_count, 1);

  batch = fill_rect(batch, 200, 300, 50, 130);
  batch = fill_rect(batch, 500, 130, 150, 100);

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_READ,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::GENERAL,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

  batch
}
