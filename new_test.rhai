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

export let file_image = allocate_image("file_image",
                                       800,
                                       680,
                                       // 1024,
                                       // 8,
                                       vk::Format::R8G8B8A8_UNORM,
                                       vk::ImageUsageFlags::SAMPLED
                                       | vk::ImageUsageFlags::TRANSFER_DST);

export let file_image_view = image_view_for(file_image);

let sampler_info =
#{ mag_filter: vk::Filter::LINEAR,
   min_filter: vk::Filter::LINEAR,
   address_mode_u: vk::SamplerAddressMode::REPEAT,
   address_mode_v: vk::SamplerAddressMode::REPEAT,
   address_mode_w: vk::SamplerAddressMode::REPEAT,
   anisotropy_enable: false,
   unnormalized_coordinates: false,
   mipmap_mode: vk::SamplerMipmapMode::NEAREST,
   mip_lod_bias: 0.0,
   min_lod: 0.0,
   max_lod: 1.0,
 };

export let sampler = create_sampler(sampler_info);


export let img_shader = load_shader("shaders/sample_image.comp.spv",
                                    vk::ShaderStageFlags::COMPUTE);


export let img_pipeline = create_compute_pipeline(img_shader);

export let img_desc_set =
  create_desc_set(img_shader,
                  0,
                  [
#{ binding: 0, sampler: sampler },
#{ binding: 1, image_view: file_image_view },
#{ binding: 2, image_view: out_view }
                  ]);


fn init() {
  let init_batch = batch_builder();

  init_batch.load_image_from_file("800px-Gradient_1-form.svg.png",
                                  self::file_image.get(),
                                  vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

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

  let pc = blob();
  pc.append_int(100);
  pc.append_int(100);

  pc.append_int(800);
  pc.append_int(860);

  pc.append_int(500);
  pc.append_int(300);

  let x_size = 8;
  let y_size = 8;

  let x_groups = (w / x_size) + w % x_size;
  let y_groups = (h / y_size) + h % y_size;


  batch.dispatch_compute(self::img_pipeline.get(),
                         self::img_desc_set.get(),
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
