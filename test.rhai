
let bindings = [vk::binding::storage_image(0),
                vk::binding::storage_buffer(1),
                vk::binding::storage_buffer(2),
                vk::binding::storage_image(3)];

export let pipeline = load_compute_shader("shaders/text_lines.comp.spv",
                                          bindings,
                                          16);

export let desc_set = desc_set_var("desc_set");

fn draw_at(x, y) {
  let batch = batch_builder();
  let p = self::pipeline.get();
  let s = self::desc_set.get();
  batch.dispatch_compute(p, s, 8, 1, 1);
  batch
}
