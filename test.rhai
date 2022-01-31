
let bindings = [vk::binding::storage_image(0),
                vk::binding::storage_buffer(1),
                vk::binding::storage_buffer(2),
                vk::binding::storage_image(3)];

export let pipeline = load_compute_shader("shaders/text_lines.comp.spv",
                                          bindings,
                                          16);

export let desc_set = desc_set_var("desc_set");

export let line_count = atomic_int("line_count");

fn draw_at(x, y) {
  let batch = batch_builder();
  let p = self::pipeline.get();
  let s = self::desc_set.get();
  let y = self::line_count.get();
  print(y);
  batch.dispatch_compute(p, s, 80, y, 1);
  batch
}
