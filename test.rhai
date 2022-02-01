
let bindings = [vk::binding::storage_image(0),
                vk::binding::storage_buffer(1),
                vk::binding::storage_buffer(2),
                vk::binding::storage_image(3)];

export let pipeline = load_compute_shader("shaders/text_lines.comp.spv",
                                          bindings,
                                          16);
export let desc_set = desc_set_var("desc_set");
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


export let out_image = image_var("out_image");

fn init() {
  let init_batch = batch_builder();
  let img = self::font_image.get();

  init_batch.load_image_from_file("8x8font.png",
                                  img,
                                  vk::ImageLayout::GENERAL);

  init_batch
}

fn draw_at(x, y) {
  let batch = batch_builder();
  let p = self::pipeline.get();
  let s = self::desc_set.get();
  let l = self::line_count.get();
  let img = self::out_image.get();

  let pc = blob();
  pc.append_int(x);
  pc.append_int(y);
  pc.append_int(800);
  pc.append_int(600);

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                         vk::ImageLayout::GENERAL);

  batch.dispatch_compute(self::pipeline.get(),
                         self::desc_set.get(),
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
