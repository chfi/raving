

export let out_image = image_var("out_image");
export let out_view = image_view_var("out_view");

export let rect_buffer =
  allocate_buffer("rect_buffer",
                  vk::MemoryLocation::CPU_TO_GPU,
                  1,
                  8 * 4 * 64, // (ivec2 + ivec2 + vec4) * 64
                  vk::BufferUsageFlags::TRANSFER_DST
                  | vk::BufferUsageFlags::STORAGE_BUFFER);


let bg_bindings = [vk::binding::storage_image(0)];
export let bg_pipeline =
  load_compute_shader("shaders/trig_color.comp.spv", bg_bindings, 8);


let bg_inputs = [#{ type: "image_view", binding: 0, view: out_view }];

export let bg_desc_set =
  create_desc_set(vk::ShaderStageFlags::COMPUTE, bg_bindings, bg_inputs);



let rect_binds = [vk::binding::storage_buffer(0),
                  vk::binding::storage_image(1)];

export let rect_pipeline =
  load_compute_shader("shaders/rects.comp.spv", rect_binds, 8);

let rect_inputs =
    [ #{ type: "buffer", binding: 0, buffer: rect_buffer },
      #{ type: "image_view", binding: 1, view: out_view } ];

export let rect_desc_set =
  create_desc_set(vk::ShaderStageFlags::COMPUTE, rect_binds, rect_inputs);

fn append_rect(buf, x, y, w, h) {
  buf.append_int(x);
  buf.append_int(y);
  buf.append_int(w);
  buf.append_int(h);

  buf.append_float(0.7);
  buf.append_float(0.6);
  buf.append_float(0.7);
  buf.append_float(1.0);

  print(buf);

  buf
}

fn init() {
  let init_batch = batch_builder();

  let rect_buf = self::rect_buffer.get();
  let rect_data = blob();

  for i in 0..5 {
    rect_data = append_rect(rect_data, 40, 50 + 100 * i, 100, 50);
    rect_data = append_rect(rect_data, 180, 40 + 100 * i, 560, 70);
  }

  init_batch.initialize_buffer_with(rect_buf, rect_data);

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

fn foreground(w, h, t) {
  let batch = batch_builder();

  let pc = blob();
  pc.append_int(w);
  pc.append_int(h);

  let img = self::out_image.get();
  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                         vk::ImageLayout::GENERAL);

  let x_size = 8;
  let y_size = 8;

  let x_groups = (w / x_size) + w % x_size;
  let y_groups = (h / y_size) + h % y_size;
  let z_groups = 1;

  batch.dispatch_compute(self::rect_pipeline.get(),
                         self::rect_desc_set.get(),
                         pc,
                         x_groups,
                         y_groups,
                         z_groups);

  // batch = fill_rect(batch, 200, 300, 50, 130);
  // batch = fill_rect(batch, 500, 130, 150, 100);

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_READ,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::GENERAL,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

  batch
}
