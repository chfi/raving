let bg_bindings = [vk::binding::storage_image(0)];

export let bg_pipeline = load_compute_shader(
  "shaders/trig_color.comp.spv",
  bg_bindings,
  8);


let label_bindings = [vk::binding::storage_image(0),
                      vk::binding::storage_buffer(1),
                      vk::binding::storage_buffer(2),
                      vk::binding::storage_image(3)];

export let label_pipeline = load_compute_shader(
  "shaders/label.comp.spv",
  label_bindings,
  8);

export let font_image = allocate_image(
  "font_image",
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

export let text_buffer = allocate_buffer(
  "text_buffer",
  vk::MemoryLocation::CPU_TO_GPU,
  1,
  256 * 256,
  vk::BufferUsageFlags::TRANSFER_DST
    | vk::BufferUsageFlags::STORAGE_BUFFER);

export let label_data_buffer =
  allocate_buffer("label_data_buffer",
                  vk::MemoryLocation::CPU_TO_GPU,
                  1,
                  256 * 4 * 4,
                  vk::BufferUsageFlags::TRANSFER_DST
                  | vk::BufferUsageFlags::STORAGE_BUFFER);


let bg_inputs = [#{ type: "image_view", binding: 0, view: out_view }];

export let bg_desc_set = create_desc_set(vk::ShaderStageFlags::COMPUTE,
                                         bg_bindings,
                                         bg_inputs);


let label_inputs = [#{ type: "image_view", binding: 0, view: font_image_view },
                    #{ type: "buffer", binding: 1, buffer: text_buffer },
                    #{ type: "buffer", binding: 2, buffer: label_data_buffer },
                    #{ type: "image_view", binding: 3, view: out_view }];


export let label_desc_set = create_desc_set(vk::ShaderStageFlags::COMPUTE,
                                           label_bindings,
                                           label_inputs);


export let label_count = atomic_int(1);



fn init() {
  let init_batch = batch_builder();
  let img = self::font_image.get();

  init_batch.load_image_from_file("8x8font.png",
                                  img,
                                  vk::ImageLayout::GENERAL);

  let text_buf = self::text_buffer.get();
  let label_buf = self::label_data_buffer.get();

  let text_data = blob();
  let label_data = blob();


  let lines = ["this is the first line",
               "here's another",
               "  . . .",
               "",
               "yes"];
}
