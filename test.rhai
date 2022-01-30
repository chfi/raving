export let pipeline = pipeline_var("pipeline");
export let desc_set = desc_set_var("desc_set");

fn draw_at_impl(x, y, batch) {
  let p = pipeline.get();
  let s = desc_set.get();

  batch.dispatch_compute(p, s, 8, 1, 1);
}
