export let out_image = image_var("out_image");
export let out_view = image_view_var("out_view");
export let out_desc_set = desc_set_var("out_desc_set");

export let bg_shader = load_shader("shaders/trig_color.comp.spv",
                                   vk::ShaderStageFlags::COMPUTE);

export let bg_pipeline = create_compute_pipeline(bg_shader);

export let bg_desc_set =
  create_desc_set(bg_shader,
                  0,
                  [ #{ binding: 0, image_view: out_view }]);

export let file_image = allocate_image("file_image",
                                       480,
                                       480,
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
                  [ #{ binding: 0, sampler: sampler },
                    #{ binding: 1, image_view: file_image_view },
                    #{ binding: 2, image_view: out_view }
                  ]);


fn init() {
  let init_batch = batch_builder();

  init_batch.load_image_from_file("yin-yang.png",
                                  self::file_image.get(),
                                  vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

  init_batch
}

fn input_pc(pos) {
  let pc = blob();
  pc.append_int(pos.x.to_int());
  pc.append_int(pos.y.to_int());

  pc.append_int(480);
  pc.append_int(480);

  pc.append_int(100);
  pc.append_int(100);

  pc
}

fn background(w, h) {
  let batch = batch_builder();
  let img = self::out_image.get();

  // transition destination image so it can be written to

  batch.transition_image(img,
                         vk::AccessFlags::empty(),
                         vk::PipelineStageFlags::TOP_OF_PIPE,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::ImageLayout::UNDEFINED,
                         vk::ImageLayout::GENERAL);

  // draw background
  let pc = blob();
  pc.append_int(w);
  pc.append_int(h);

  let x_size = 16;
  let y_size = 16;

  let x_groups = (w / x_size) + w % x_size;
  let y_groups = (h / y_size) + h % y_size;

  batch.dispatch_compute(self::bg_pipeline.get(),
                         self::bg_desc_set.get(),
                         pc, x_groups, y_groups, 1);

  // workgroup counts for image copy
  let x_size = 8;
  let y_size = 8;

  let x_groups = (w / x_size) + w % x_size;
  let y_groups = (h / y_size) + h % y_size;

  let group_counts =
    #{ x_groups: (w / x_size) + w % x_size,
       y_groups: (h / y_size) + h % y_size,
       z_groups: 1 };

  // loop that draws each image
  for i in 0..6 {
    let fi = i.to_float();
    let theta = (fi / 6) * 2.0 * PI();

    let radius = 150.0;

    let img_pos = #{ x: 300.0 + radius * theta.cos(),
                     y: 200.0 + radius * theta.sin() };

    let push_constant = input_pc(img_pos);

    batch.dispatch_compute(self::img_pipeline.get(),
                           [self::img_desc_set.get()],
                           push_constant,
                           group_counts);
  }

  // transition destination image so it can be copied from

  batch.transition_image(img,
                         vk::AccessFlags::SHADER_WRITE,
                         vk::PipelineStageFlags::COMPUTE_SHADER,
                         vk::AccessFlags::TRANSFER_READ,
                         vk::PipelineStageFlags::TRANSFER,
                         vk::ImageLayout::GENERAL,
                         vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

  batch
}
