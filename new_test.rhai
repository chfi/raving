export let out_image = image_var("out_image");
export let out_view = image_view_var("out_view");

export let rect_buffer =
  allocate_buffer("rect_buffer",
                  vk::MemoryLocation::CPU_TO_GPU,
                  1,
                  8 * 4 * 64, // (ivec2 + ivec2 + vec4) * 64
                  vk::BufferUsageFlags::TRANSFER_DST
                  | vk::BufferUsageFlags::STORAGE_BUFFER);

export let bg_shader = load_shader("shaders/trig_color.comp.spv",
                                   vk::ShaderStageFlags::COMPUTE);

export let bg_pipeline = create_compute_pipeline(bg_shader);

export let bg_desc_set =
  create_desc_set(bg_shader,
                  0,
                  [ #{ binding: 0, image_view: out_view }]);

fn init() {
  let init_batch = batch_builder();
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
