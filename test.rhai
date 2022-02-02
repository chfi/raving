
let bg_bindings = [vk::binding::storage_image(0)];
export let bg_pipeline = load_compute_shader("shaders/trig_color.comp.spv",
                                      bg_bindings,
                                      8);

export let bg_desc_set = desc_set_var("bg_desc_set");

let line_bindings = [vk::binding::storage_image(0),
                     vk::binding::storage_buffer(1),
                     vk::binding::storage_buffer(2),
                     vk::binding::storage_image(3)];

export let line_pipeline = load_compute_shader("shaders/text_lines.comp.spv",
                                               line_bindings,
                                               16);
export let line_desc_set = desc_set_var("line_desc_set");
export let line_count = atomic_int("line_count");


export let font_image = allocate_image("font_image",
                                       1024,
                                       8,
                                       vk::Format::R8G8B8A8_UNORM,
                                       vk::ImageUsageFlags::STORAGE
                                       | vk::ImageUsageFlags::SAMPLED
                                       | vk::ImageUsageFlags::TRANSFER_SRC
                                       | vk::ImageUsageFlags::TRANSFER_DST);

export let font_image_view = image_view_for(font_image);


export let bg_image = image_var("bg_image");
export let out_image = image_var("out_image");
export let out_view = image_var("out_view");

export let text_buffer = buffer_var("text_buffer");
export let line_buffer = buffer_var("line_buffer");

let line_inputs = [{ type: "image_view", binding: 0, view: font_image_view },
                   { type: "buffer", binding: 1, buffer: text_buffer },
                   { type: "buffer", binding: 2, buffer: line_buffer },
                   { type: "image_view", binding: 3, view: out_view }];


let line_desc_set_ = create_desc_set(vk::ShaderStageFlags::COMPUTE,
                                     line_bindings,
                                     line_inputs);


fn init() {
  let init_batch = batch_builder();
  let img = self::font_image.get();

  init_batch.load_image_from_file("8x8font.png",
                                  img,
                                  vk::ImageLayout::GENERAL);

  init_batch
}

fn background(w, h) {
  let batch = batch_builder();
  let img = self::bg_image.get();

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
  let l = self::line_count.get();
  let img = self::out_image.get();

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
                         pc, 80, l, 1);

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_READ,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::GENERAL,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

  batch
}
