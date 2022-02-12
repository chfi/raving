

export let out_image = image_var("out_image");
export let out_view = image_view_var("out_view");


export let font_image = allocate_image("font_image",
                                       1024,
                                       8,
                                       vk::Format::R8G8B8A8_UNORM,
                                       vk::ImageUsageFlags::STORAGE
                                       | vk::ImageUsageFlags::SAMPLED
                                       | vk::ImageUsageFlags::TRANSFER_SRC
                                       | vk::ImageUsageFlags::TRANSFER_DST);

export let font_image_view = image_view_for(font_image);


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
  // load_compute_shader("shaders/rects.comp.spv.old", rect_binds, 8);
  load_compute_shader("shaders/rects.comp.spv", rect_binds, 8);

let rect_inputs =
    [ #{ type: "buffer", binding: 0, buffer: rect_buffer },
      #{ type: "image_view", binding: 1, view: out_view } ];

export let rect_desc_set =
  create_desc_set(vk::ShaderStageFlags::COMPUTE, rect_binds, rect_inputs);


let line_bindings = [vk::binding::storage_image(0),
                     vk::binding::storage_buffer(1),
                     vk::binding::storage_buffer(2),
                     vk::binding::storage_image(3)];

export let line_pipeline = load_compute_shader("shaders/text_lines.comp.spv",
                                               line_bindings,
                                               16);


export let line_desc_sets = [];

export let text_buffers = [];
export let line_buffers = [];

for i in 0..5 {
  let text = allocate_buffer("text_buffer",
                             vk::MemoryLocation::CPU_TO_GPU,
                             1,
                             256 * 256,
                             vk::BufferUsageFlags::TRANSFER_DST
                             | vk::BufferUsageFlags::STORAGE_BUFFER);

  let line = allocate_buffer("line_buffer",
                             vk::MemoryLocation::CPU_TO_GPU,
                             1,
                             256 * 4 * 2,
                             vk::BufferUsageFlags::TRANSFER_DST
                             | vk::BufferUsageFlags::STORAGE_BUFFER);

  text_buffers.push(text);
  line_buffers.push(line);

  let line_inputs = [#{ type: "image_view", binding: 0, view: font_image_view },
                     #{ type: "buffer", binding: 1, buffer: text },
                     #{ type: "buffer", binding: 2, buffer: line },
                     #{ type: "image_view", binding: 3, view: out_view }];


  let set = create_desc_set(vk::ShaderStageFlags::COMPUTE,
                            line_bindings,
                            line_inputs);

  line_desc_sets.push(set);

}


fn rect_blob(rect) {
  let buf = blob();
  buf.append_int(rect.x);
  buf.append_int(rect.y);
  buf.append_int(rect.w);
  buf.append_int(rect.h);

  buf.append_float(0.6);
  buf.append_float(0.55);
  buf.append_float(0.6);
  buf.append_float(1.0);
  buf
}

fn init() {
  let init_batch = batch_builder();


  init_batch.load_image_from_file("8x8font.png",
                                  self::font_image.get(),
                                  vk::ImageLayout::GENERAL);

  let rect_buf = self::rect_buffer.get();
  let rect_data = blob();

  let text_bufs = self::text_buffers;
  let line_bufs = self::line_buffers;

  let prefix = "label ";

  let rects = [];

  for i in 0..5 {
    let line = prefix + i.to_string();

    let text_buf = text_bufs[i].get();
    let line_buf = line_bufs[i].get();

    let text_data = blob();
    let line_data = blob();

    text_data.append_int(line.len());
    text_data.append_str_tmp(line);

    line_data.append_int(0);
    line_data.append_int(line.len());

    init_batch.initialize_buffer_with(text_buf, text_data);
    init_batch.initialize_buffer_with(line_buf, line_data);

    rects.push(#{ x: 40,  y: 50 + 100 * i, w: 100, h: 50 });
    rects.push(#{ x: 180, y: 40 + 100 * i, w: 560, h: 70 });
  }

  init_batch.initialize_buffer_with(Fn("rect_blob"), rect_buf, rects);

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

  let rect_pc = blob();
  rect_pc.append_int(w);
  rect_pc.append_int(h);

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

  let z_groups = 5 * 2;

  batch.dispatch_compute(self::rect_pipeline.get(),
                         self::rect_desc_set.get(),
                         rect_pc,
                         x_groups,
                         y_groups,
                         z_groups);


  for i in 0..5 {
    let set = self::line_desc_sets[i].get();

    let x = 50;
    let y = 71 + 100 * i;

    let pc = blob();
    pc.append_int(x.to_int());
    pc.append_int(y.to_int());
    pc.append_int(w);
    pc.append_int(h);

    batch.dispatch_compute(self::line_pipeline.get(),
                           set,
                           pc,
                           80,
                           1,
                           1);
  }

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_READ,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::GENERAL,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

  batch
}
