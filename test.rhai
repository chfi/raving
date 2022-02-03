
let bg_bindings = [vk::binding::storage_image(0)];
export let bg_pipeline = load_compute_shader("shaders/trig_color.comp.spv",
                                      bg_bindings,
                                      8);

let line_bindings = [vk::binding::storage_image(0),
                     vk::binding::storage_buffer(1),
                     vk::binding::storage_buffer(2),
                     vk::binding::storage_image(3)];

export let line_pipeline = load_compute_shader("shaders/text_lines.comp.spv",
                                               line_bindings,
                                               16);

export let font_image = allocate_image("font_image",
                                       1024,
                                       8,
                                       vk::Format::R8G8B8A8_UNORM,
                                       vk::ImageUsageFlags::STORAGE
                                       | vk::ImageUsageFlags::SAMPLED
                                       | vk::ImageUsageFlags::TRANSFER_SRC
                                       | vk::ImageUsageFlags::TRANSFER_DST);

export let font_image_view = image_view_for(font_image);

export let out_image = image_var("out_image");
export let out_view = image_view_var("out_view");

export let text_buffer = allocate_buffer("text_buffer",
                                         vk::MemoryLocation::CPU_TO_GPU,
                                         1,
                                         256 * 256,
                                         vk::BufferUsageFlags::TRANSFER_DST
                                         | vk::BufferUsageFlags::STORAGE_BUFFER);

export let line_buffer = allocate_buffer("line_buffer",
                                         vk::MemoryLocation::CPU_TO_GPU,
                                         1,
                                         256 * 4 * 2,
                                         vk::BufferUsageFlags::TRANSFER_DST
                                         | vk::BufferUsageFlags::STORAGE_BUFFER);

let bg_inputs = [#{ type: "image_view", binding: 0, view: out_view }];

export let bg_desc_set = create_desc_set(vk::ShaderStageFlags::COMPUTE,
                                         bg_bindings,
                                         bg_inputs);


let line_inputs = [#{ type: "image_view", binding: 0, view: font_image_view },
                   #{ type: "buffer", binding: 1, buffer: text_buffer },
                   #{ type: "buffer", binding: 2, buffer: line_buffer },
                   #{ type: "image_view", binding: 3, view: out_view }];


export let line_desc_set = create_desc_set(vk::ShaderStageFlags::COMPUTE,
                                           line_bindings,
                                           line_inputs);

export let line_count = atomic_int(1);


fn init() {
  let init_batch = batch_builder();
  let img = self::font_image.get();

  init_batch.load_image_from_file("8x8font.png",
                                  img,
                                  vk::ImageLayout::GENERAL);

  let text_buf = self::text_buffer.get();
  let line_buf = self::line_buffer.get();

  let text_data = blob();
  let line_data = blob();


  let lines = ["this is the first line",
               "here's another",
               "  . . .",
               "",
               "yes"];

  let offset = 0;

  text_data.append_int(0);

  self::line_count.set(lines.len());

  for line in lines {
    line_data.append_int(offset);
    offset += line.len();
    line_data.append_int(offset);

    text_data.append_str_tmp(line);
  }

  text_data.write_le(0, 4, offset);

  // buf.append_str_tmp("wha the fucklacoehu");
  // print(buf);
  init_batch.initialize_buffer_with(text_buf, text_data);
  init_batch.initialize_buffer_with(line_buf, line_data);

  init_batch
}

fn background(w, h) {
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

fn draw_at(x, y, w, h) {
  let batch = batch_builder();
  let img = self::out_image.get();
  let line_count = self::line_count.get();

  let pc = blob();
  pc.append_int(x);
  pc.append_int(y);
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

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_READ,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::GENERAL,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

  batch
}
