
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
  let l = self::line_count.get();

  let pc = blob();

  pc.append_int(x);
  pc.append_int(y);
  pc.append_int(800);
  pc.append_int(600);

  batch.dispatch_compute(p, s, pc, 80, l, 1);
  batch
}
