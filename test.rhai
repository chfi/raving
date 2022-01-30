export let pipeline = pipeline_var("pipeline");
export let desc_set = desc_set_var("desc_set");

fn draw_at(x, y) {
  let batch = batch_builder();
  let p = self::pipeline.get();
  let s = self::desc_set.get();
  batch.dispatch_compute(p, s, 8, 1, 1);
  batch
}
