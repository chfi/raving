use ash::vk;
use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use parking_lot::{Mutex, RwLock, RwLockReadGuard};

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use std::sync::Arc;

use crate::vk::{
    context::VkContext, descriptor::BindingDesc, resource::index::*,
    util::LineRenderer, GpuResources, VkEngine,
};

use rhai::plugin::*;

use super::vk as rvk;

pub type BatchFn =
    Arc<dyn Fn(&ash::Device, &GpuResources, vk::CommandBuffer) + Send + Sync>;

pub type InitFn = Arc<
    dyn Fn(
            &VkContext,
            &mut GpuResources,
            &mut Allocator,
            vk::CommandBuffer,
        ) -> anyhow::Result<()>
        + Send
        + Sync,
>;

#[derive(Default, Clone)]
pub struct BatchBuilder {
    pub init_fn: Vec<InitFn>,

    staging_buffers: Arc<Mutex<Vec<BufferIx>>>,

    command_fns: Vec<BatchFn>,
    // Vec<Box<dyn Fn(&ash::Device, &GpuResources, vk::CommandBuffer)>>,
}

impl BatchBuilder {
    pub fn build(self) -> BatchFn {
        let cmds = Arc::new(self.command_fns);

        let batch =
            Arc::new(move |dev: &ash::Device, res: &GpuResources, cmd| {
                let cmds = cmds.clone();

                for f in cmds.iter() {
                    f(dev, res, cmd);
                }
            }) as BatchFn;

        batch
    }

    pub fn free_staging_buffers(
        &mut self,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> anyhow::Result<()> {
        let mut buffers = self.staging_buffers.lock();

        for buf_ix in buffers.drain(..) {
            res.free_buffer(ctx, alloc, buf_ix)?;
        }

        Ok(())
    }

    pub fn load_image_from_file(
        &mut self,
        file_path: &str,
        dst_image: ImageIx,
        final_layout: vk::ImageLayout,
    ) {
        let file_path = file_path.to_string();

        let staging_bufs = self.staging_buffers.clone();

        let f = Arc::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator,
                  cmd: vk::CommandBuffer| {
                let dev = ctx.device();

                let img = &mut res[dst_image];
                let vk_img = img.image;

                VkEngine::transition_image(
                    cmd,
                    dev,
                    vk_img,
                    vk::AccessFlags::empty(),
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );

                use image::io::Reader as ImageReader;

                let font = ImageReader::open(&file_path)?.decode()?;

                let font_rgba8 = font.to_rgba8();

                let pixel_bytes =
                    font_rgba8.enumerate_pixels().flat_map(|(_, _, col)| {
                        let [r, g, b, a] = col.0;
                        [r, g, b, a].into_iter()
                    });

                let staging = img.fill_from_pixels(
                    dev,
                    ctx,
                    alloc,
                    pixel_bytes,
                    4,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    cmd,
                )?;

                let staging_ix = res.insert_buffer(staging);

                staging_bufs.lock().push(staging_ix);

                VkEngine::transition_image(
                    cmd,
                    dev,
                    vk_img,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::MEMORY_READ,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout,
                );

                Ok(())
            },
        ) as InitFn;

        self.init_fn.push(f);
    }

    pub fn transition_image(
        &mut self,
        image: ImageIx,
        src_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_access_mask: vk::AccessFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let f = Arc::new(move |dev: &ash::Device, res: &GpuResources, cmd| {
            let img = &res[image];

            VkEngine::transition_image(
                cmd,
                dev,
                img.image,
                src_access_mask,
                src_stage_mask,
                dst_access_mask,
                dst_stage_mask,
                old_layout,
                new_layout,
            );
        }) as BatchFn;

        self.command_fns.push(f);
    }

    pub fn dispatch_compute(
        &mut self,
        pipeline: PipelineIx,
        desc_set: DescSetIx,
        push_constants: Vec<u8>,
        x_groups: i64,
        y_groups: i64,
        z_groups: i64,
    ) {
        let bytes = Arc::new(push_constants);

        let inner = bytes.clone();
        let f = Arc::new(move |dev: &ash::Device, res: &GpuResources, cmd| {
            let bytes = inner.clone();

            let groups = (x_groups as u32, y_groups as u32, z_groups as u32);

            VkEngine::dispatch_compute(
                res,
                dev,
                cmd,
                pipeline,
                desc_set,
                bytes.as_slice(),
                groups,
            );
        }) as BatchFn;

        self.command_fns.push(f);
    }
}

#[derive(Default)]
pub struct ModuleBuilder {
    resolvers: [Vec<WithAllocators<()>>; 10],

    image_resolvers: Vec<WithAllocators<()>>,
    image_view_resolvers: Vec<WithAllocators<()>>,

    buffer_resolvers: Vec<WithAllocators<()>>,
    pipeline_resolvers: Vec<WithAllocators<()>>,

    desc_set_resolvers: Vec<WithAllocators<()>>,

    image_vars: FxHashMap<String, Resolvable<ImageIx>>,
    image_view_vars: FxHashMap<String, Resolvable<ImageViewIx>>,

    buffer_vars: FxHashMap<String, Resolvable<BufferIx>>,
    pipeline_vars: FxHashMap<String, Resolvable<PipelineIx>>,

    desc_set_vars: FxHashMap<String, Resolvable<DescSetIx>>,

    ints: FxHashMap<String, Arc<AtomicCell<i64>>>,
    floats: FxHashMap<String, Arc<AtomicCell<f32>>>,

    // images: FxHashMap<usize, Resolvable<ImageIx>>,
    // rhai_mod: rhai::Module,
    pub script_ast: rhai::AST,
}

pub fn create_batch_engine() -> rhai::Engine {
    let mut engine = rhai::Engine::new();

    engine.register_type_with_name::<ash::vk::Format>("Format");
    engine
        .register_type_with_name::<ash::vk::ImageUsageFlags>("ImageUsageFlags");
    engine.register_type_with_name::<ash::vk::BufferUsageFlags>(
        "BufferUsageFlags",
    );

    let vk_mod = rhai::exported_module!(super::vk);
    engine.register_static_module("vk", vk_mod.into());

    engine.register_type_with_name::<ImageIx>("ImageIx");
    engine.register_type_with_name::<ImageViewIx>("ImageViewIx");
    engine.register_type_with_name::<BufferIx>("BufferIx");
    engine.register_type_with_name::<PipelineIx>("PipelineIx");
    engine.register_type_with_name::<DescSetIx>("DescSetIx");

    engine
        .register_type_with_name::<Resolvable<ImageIx>>("Resolvable<ImageIx>");
    engine.register_type_with_name::<Resolvable<ImageViewIx>>(
        "Resolvable<ImageViewIx>",
    );
    engine.register_type_with_name::<Resolvable<BufferIx>>(
        "Resolvable<BufferIx>",
    );
    engine.register_type_with_name::<Resolvable<PipelineIx>>(
        "Resolvable<PipelineIx>",
    );
    engine.register_type_with_name::<Resolvable<DescSetIx>>(
        "Resolvable<DescSetIx>",
    );

    engine.register_type_with_name::<BatchBuilder>("BatchBuilder");

    engine.register_fn("get", |img: Resolvable<ImageIx>| {
        img.value.load().unwrap()
    });
    engine.register_fn("get", |view: Resolvable<ImageViewIx>| {
        view.value.load().unwrap()
    });
    engine.register_fn("get", |buf: Resolvable<BufferIx>| {
        buf.value.load().unwrap()
    });
    engine.register_fn("get", |pipeline: Resolvable<PipelineIx>| {
        pipeline.value.load().unwrap()
    });
    engine.register_fn("get", |set: Resolvable<DescSetIx>| {
        set.value.load().unwrap()
    });

    engine.register_fn("get", |v: Arc<AtomicCell<i64>>| v.load());
    engine.register_fn("get", |v: Arc<AtomicCell<f32>>| v.load());

    engine.register_fn("append_int", |blob: &mut Vec<u8>, v: i64| {
        let v = [v as i32];
        blob.extend_from_slice(bytemuck::cast_slice(&v));
    });

    engine.register_fn("append_float", |blob: &mut Vec<u8>, v: f32| {
        let v = [v as f32];
        blob.extend_from_slice(bytemuck::cast_slice(&v));
    });

    engine.register_fn("batch_builder", || BatchBuilder::default());

    engine.register_fn(
        "load_image_from_file",
        |builder: &mut BatchBuilder,
         file_path: &str,
         dst_image: ImageIx,
         final_layout: vk::ImageLayout| {
            builder.load_image_from_file(file_path, dst_image, final_layout)
        },
    );

    engine.register_fn(
        "transition_image",
        |builder: &mut BatchBuilder,
         image: ImageIx,
         src_access_mask: vk::AccessFlags,
         src_stage_mask: vk::PipelineStageFlags,
         dst_access_mask: vk::AccessFlags,
         dst_stage_mask: vk::PipelineStageFlags,
         old_layout: vk::ImageLayout,
         new_layout: vk::ImageLayout| {
            builder.transition_image(
                image,
                src_access_mask,
                src_stage_mask,
                dst_access_mask,
                dst_stage_mask,
                old_layout,
                new_layout,
            );
        },
    );

    engine.register_fn(
        "dispatch_compute",
        |builder: &mut BatchBuilder,
         pipeline: PipelineIx,
         desc_set: DescSetIx,
         push_constants: Vec<u8>,
         x_groups: i64,
         y_groups: i64,
         z_groups: i64| {
            builder.dispatch_compute(
                pipeline,
                desc_set,
                push_constants,
                x_groups,
                y_groups,
                z_groups,
            );
        },
    );

    engine
}

impl ModuleBuilder {
    pub fn int_cell(&self, k: &str) -> Option<&Arc<AtomicCell<i64>>> {
        self.ints.get(k)
    }

    pub fn float_cell(&self, k: &str) -> Option<&Arc<AtomicCell<f32>>> {
        self.floats.get(k)
    }

    pub fn set_int(&mut self, k: &str, v: i64) {
        let cell = self.ints.get(k).unwrap();
        cell.store(v);
    }

    pub fn set_float(&mut self, k: &str, v: f32) {
        let cell = self.floats.get(k).unwrap();
        cell.store(v);
    }

    pub fn from_script(path: &str) -> anyhow::Result<(Self, rhai::Module)> {
        let (module, ast, arcres) = {
            let mut engine = create_batch_engine();

            let result = Self::default();
            let arcres = Arc::new(Mutex::new(result));

            let res = arcres.clone();
            engine.register_fn(
                "allocate_image",
                move |name: &str,
                      width: i64,
                      height: i64,
                      format: vk::Format,
                      usage: vk::ImageUsageFlags| {
                    let resolvable = res.lock().allocate_image(
                        width as u32,
                        height as u32,
                        format,
                        usage,
                        Some(name),
                    );
                    resolvable
                },
            );

            let res = arcres.clone();
            engine.register_fn(
                "image_view_for",
                move |image: Resolvable<ImageIx>| {
                    let resolvable = res.lock().create_image_view(image);
                    resolvable
                },
            );

            let res = arcres.clone();
            engine.register_fn(
                "load_compute_shader",
                move |path: &str, bindings: rhai::Array, pc_size: i64| {
                    let bindings = bindings
                        .into_iter()
                        .map(|b| b.cast::<BindingDesc>())
                        .collect::<Vec<_>>();

                    let resolvable = res.lock().load_compute_shader(
                        path,
                        &bindings,
                        pc_size as usize,
                    );
                    resolvable
                },
            );

            /*
            let res = arcres.clone();
            engine.register_fn(
                "input_image_view",
                move |binding: i64,
                      view: Resolvable<ImageViewIx>|
                      -> Resolvable<BindingInput> {
                    //
                },
            );
            */

            let res = arcres.clone();
            engine.register_fn("atomic_int", move |name: &str| {
                let var = Arc::new(AtomicCell::new(0i64));
                res.lock().ints.insert(name.to_string(), var.clone());
                var
            });

            let res = arcres.clone();
            engine.register_fn("atomic_float", move |name: &str| {
                let var = Arc::new(AtomicCell::new(0f32));
                res.lock().floats.insert(name.to_string(), var.clone());
                var
            });

            let res = arcres.clone();
            engine.register_fn("image_var", move |name: &str| {
                let resolvable = res.lock().image_var(name);
                resolvable
            });

            let res = arcres.clone();
            engine.register_fn("image_view_var", move |name: &str| {
                let resolvable = res.lock().image_view_var(name);
                resolvable
            });

            let res = arcres.clone();
            engine.register_fn("buffer_var", move |name: &str| {
                let resolvable = res.lock().buffer_var(name);
                resolvable
            });

            let res = arcres.clone();
            engine.register_fn("pipeline_var", move |name: &str| {
                let resolvable = res.lock().pipeline_var(name);
                resolvable
            });

            let res = arcres.clone();
            engine.register_fn("desc_set_var", move |name: &str| {
                let resolvable = res.lock().desc_set_var(name);
                resolvable
            });

            let path = std::path::PathBuf::from(path);
            let ast = engine.compile_file(path)?;

            let module = rhai::Module::eval_ast_as_new(
                rhai::Scope::new(),
                &ast,
                &engine,
            )?;

            let _ = engine;

            // let result = Arc::

            (module, ast, arcres)
        };

        let mutex = Arc::try_unwrap(arcres).ok().unwrap();
        let mut result = mutex.into_inner();

        result.script_ast = ast;
        // module.build_index()

        for (name, var) in module.iter_var() {
            log::warn!("{} - {:?}", name, var);
        }

        Ok((result, module))
    }

    pub fn image_var(&mut self, k: &str) -> Resolvable<ImageIx> {
        if let Some(res) = self.image_vars.get(k) {
            res.clone()
        } else {
            let resolvable = Resolvable::empty();
            self.image_vars.insert(k.to_string(), resolvable.clone());
            resolvable
        }
    }

    pub fn image_view_var(&mut self, k: &str) -> Resolvable<ImageViewIx> {
        if let Some(res) = self.image_view_vars.get(k) {
            res.clone()
        } else {
            let resolvable = Resolvable::empty();
            self.image_view_vars
                .insert(k.to_string(), resolvable.clone());
            resolvable
        }
    }

    pub fn buffer_var(&mut self, k: &str) -> Resolvable<BufferIx> {
        if let Some(res) = self.buffer_vars.get(k) {
            res.clone()
        } else {
            let resolvable = Resolvable::empty();
            self.buffer_vars.insert(k.to_string(), resolvable.clone());
            resolvable
        }
    }

    pub fn pipeline_var(&mut self, k: &str) -> Resolvable<PipelineIx> {
        if let Some(res) = self.pipeline_vars.get(k) {
            res.clone()
        } else {
            let resolvable = Resolvable::empty();
            self.pipeline_vars.insert(k.to_string(), resolvable.clone());
            resolvable
        }
    }

    pub fn desc_set_var(&mut self, k: &str) -> Resolvable<DescSetIx> {
        if let Some(res) = self.desc_set_vars.get(k) {
            res.clone()
        } else {
            let resolvable = Resolvable::empty();
            self.desc_set_vars.insert(k.to_string(), resolvable.clone());
            resolvable
        }
    }

    pub fn bind_image_var(&mut self, k: &str, v: ImageIx) -> Option<()> {
        let var = self
            .image_vars
            .remove(k)
            .expect("tried to bind to missing variable");
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_image_view_var(
        &mut self,
        k: &str,
        v: ImageViewIx,
    ) -> Option<()> {
        let var = self
            .image_view_vars
            .remove(k)
            .expect("tried to bind to missing variable");
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_buffer_var(&mut self, k: &str, v: BufferIx) -> Option<()> {
        let var = self
            .buffer_vars
            .remove(k)
            .expect("tried to bind to missing variable");
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_pipeline_var(&mut self, k: &str, v: PipelineIx) -> Option<()> {
        let var = self
            .pipeline_vars
            .remove(k)
            .expect("tried to bind to missing variable");
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_desc_set_var(&mut self, k: &str, v: DescSetIx) -> Option<()> {
        let var = self
            .desc_set_vars
            .remove(k)
            .expect("tried to bind to missing variable");
        var.value.store(Some(v));
        Some(())
    }

    pub fn resolve(
        &mut self,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> anyhow::Result<bool> {
        for f in self.pipeline_resolvers.drain(..) {
            f(ctx, res, alloc)?;
        }

        for f in self.image_resolvers.drain(..) {
            f(ctx, res, alloc)?;
        }

        for f in self.image_view_resolvers.drain(..) {
            f(ctx, res, alloc)?;
        }

        for f in self.buffer_resolvers.drain(..) {
            f(ctx, res, alloc)?;
        }

        for f in self.desc_set_resolvers.drain(..) {
            f(ctx, res, alloc)?;
        }

        Ok(self.is_resolved())
    }

    pub fn is_resolved(&self) -> bool {
        let resolvers = self.image_resolvers.is_empty()
            && self.image_view_resolvers.is_empty()
            && self.buffer_resolvers.is_empty()
            && self.pipeline_resolvers.is_empty()
            && self.desc_set_resolvers.is_empty();

        let vars = self.image_vars.is_empty()
            && self.image_view_vars.is_empty()
            && self.buffer_vars.is_empty()
            && self.pipeline_vars.is_empty()
            && self.desc_set_vars.is_empty();

        log::warn!("resolvers = {}\tvars = {}", resolvers, vars);

        resolvers && vars
    }

    pub fn create_image_view(
        &mut self,
        image_ix: Resolvable<ImageIx>,
    ) -> Resolvable<ImageViewIx> {
        let resolvable = Arc::new(AtomicCell::new(None));

        let inner = resolvable.clone();

        let resolver = Box::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                let image = image_ix.value.load().unwrap();
                let view = res.create_image_view_for_image(ctx, image)?;
                inner.store(Some(view));
                Ok(())
            },
        ) as WithAllocators<()>;

        self.image_view_resolvers.push(resolver);

        Resolvable::new(resolvable)
    }

    pub fn allocate_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<&str>,
    ) -> Resolvable<ImageIx> {
        let resolvable = Arc::new(AtomicCell::new(None));

        let name = name.map(String::from);

        let inner = resolvable.clone();

        let resolver = Box::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                let img = res.allocate_image(
                    ctx,
                    alloc,
                    width,
                    height,
                    format,
                    usage,
                    name.as_ref().map(|s| s.as_str()),
                )?;
                inner.store(Some(img));
                Ok(())
            },
        ) as WithAllocators<()>;

        self.image_resolvers.push(resolver);

        Resolvable::new(resolvable)
    }

    pub fn load_compute_shader(
        &mut self,
        shader_path: &str,
        bindings: &[BindingDesc],
        pc_size: usize,
    ) -> Resolvable<PipelineIx> {
        let resolvable = Arc::new(AtomicCell::new(None));

        let inner = resolvable.clone();

        let shader_path = shader_path.to_string();
        let bindings = Vec::from(bindings);

        let resolver = Box::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  _alloc: &mut Allocator| {
                let pipeline = res.load_compute_shader_runtime(
                    ctx,
                    &shader_path,
                    &bindings,
                    pc_size,
                )?;
                inner.store(Some(pipeline));
                Ok(())
            },
        ) as WithAllocators<()>;

        self.pipeline_resolvers.push(resolver);

        Resolvable::new(resolvable)
    }
}

// pub enum ResolvePriority {
//     Buffer = 0,
// }

#[derive(Clone)]
pub struct Resolvable<T: Copy> {
    priority: u64,
    value: Arc<AtomicCell<Option<T>>>,
}

impl<T: Copy> Resolvable<T> {
    pub fn new(value: Arc<AtomicCell<Option<T>>>) -> Self {
        Self { value, priority: 0 }
    }

    pub fn empty() -> Self {
        Self {
            priority: 10,
            value: Arc::new(AtomicCell::new(None)),
        }
    }

    pub fn get(&self) -> Option<T> {
        self.value.load()
    }
}

#[derive(Clone, Default)]
pub struct ResolvableRef<T> {
    value: Arc<RwLock<Option<T>>>,
}

impl<T> ResolvableRef<T> {
    pub fn get(&self) -> Option<RwLockReadGuard<Option<T>>> {
        let lock = self.value.try_read()?;
        if lock.is_none() {
            return None;
        }
        Some(lock)
    }
}

pub type WithAllocators<T> = Box<
    dyn FnOnce(
            &VkContext,
            &mut GpuResources,
            &mut Allocator,
        ) -> anyhow::Result<T>
        + Send
        + Sync,
>;

pub mod resolver {

    use crossbeam::atomic::AtomicCell;
    use gpu_allocator::vulkan::Allocator;
    use std::{collections::BTreeMap, sync::Arc};

    use crate::vk::resource::index::*;
    use crate::vk::{context::VkContext, GpuResources};

    use super::WithAllocators;

    pub struct Resolvable<T: Copy> {
        priority: Priority,
        value: Arc<AtomicCell<Option<T>>>,
    }

    impl<T: Copy> Resolvable<T> {
        pub fn get(&self) -> Option<T> {
            self.value.load()
        }

        pub fn priority(&self) -> Priority {
            self.priority
        }

        pub fn priority_type(&self) -> ResolveType {
            self.priority.ty
        }
    }

    pub struct Resolvers {
        resolvers: BTreeMap<Priority, Vec<WithAllocators<()>>>,
    }

    impl Resolvers {
        pub fn and_then<
            T: Copy + Send + Sync + 'static,
            U: Copy + Send + Sync + 'static,
        >(
            &mut self,
            f: impl FnOnce(T) -> U + Send + Sync + 'static,
            r: Resolvable<T>,
        ) -> Resolvable<U> {
            let priority = Priority {
                ty: r.priority_type(),
                secondary: true,
            };

            let cell = Arc::new(AtomicCell::new(None));
            let inner = cell.clone();

            let resolver = Box::new(
                move |_: &VkContext,
                      _: &mut GpuResources,
                      _: &mut Allocator| {
                    let t = r.get().unwrap();
                    let u = f(t);
                    inner.store(Some(u));
                    Ok(())
                },
            ) as WithAllocators<()>;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn create_image_view(
            &mut self,
            image_ix: Resolvable<ImageIx>,
            name: Option<&str>,
        ) -> Resolvable<ImageViewIx> {
            let priority = Priority {
                ty: ResolveType::ImageView,
                secondary: false,
            };

            let cell = Arc::new(AtomicCell::new(None));
            // let name = name.map(String::from);
            let inner = cell.clone();

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      _: &mut Allocator| {
                    let image = image_ix.value.load().unwrap();
                    let view = res.create_image_view_for_image(ctx, image)?;
                    inner.store(Some(view));
                    Ok(())
                },
            ) as WithAllocators<()>;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn allocate_image(
            &mut self,
            width: u32,
            height: u32,
            format: ash::vk::Format,
            usage: ash::vk::ImageUsageFlags,
            name: Option<&str>,
        ) -> Resolvable<ImageIx> {
            let priority = Priority {
                ty: ResolveType::Image,
                secondary: false,
            };

            let cell = Arc::new(AtomicCell::new(None));
            let name = name.map(String::from);
            let inner = cell.clone();

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      alloc: &mut Allocator| {
                    let img = res.allocate_image(
                        ctx,
                        alloc,
                        width,
                        height,
                        format,
                        usage,
                        name.as_ref().map(|s| s.as_str()),
                    )?;
                    inner.store(Some(img));
                    Ok(())
                },
            ) as WithAllocators<()>;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(u64)]
    pub enum ResolveType {
        Buffer = 0,
        Image = 1,
        ImageView = 2,
        Sampler = 3,
        SampledImage = 4,
        BindingInput = 5,
        DescriptorSet = 6,
        Other = 7,
    }

    #[derive(Clone, Copy)]
    pub struct Priority {
        ty: ResolveType,
        secondary: bool,
    }

    impl Priority {
        pub const fn as_primary(&self) -> Self {
            Self {
                secondary: false,
                ..*self
            }
        }

        pub const fn as_secondary(&self) -> Self {
            Self {
                secondary: true,
                ..*self
            }
        }

        pub const fn as_i32(&self) -> i32 {
            let i = self.ty as i32;
            (i << 1) | self.secondary as i32
        }

        pub const fn as_u64(&self) -> u64 {
            let i = self.ty as u64;
            (i << 1) | self.secondary as u64
        }
    }

    impl PartialEq for Priority {
        fn eq(&self, other: &Priority) -> bool {
            self.as_u64() == other.as_u64()
        }
    }

    impl Eq for Priority {}

    impl PartialOrd for Priority {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.as_u64().partial_cmp(&other.as_u64())
        }
    }

    impl Ord for Priority {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.as_u64().cmp(&other.as_u64())
        }
    }
}

pub mod frame {

    use crossbeam::atomic::AtomicCell;
    use gpu_allocator::vulkan::Allocator;
    use parking_lot::Mutex;
    use rustc_hash::FxHashMap;
    use std::any::TypeId;
    use std::collections::HashMap;
    use std::{collections::BTreeMap, sync::Arc};

    use crate::vk::descriptor::{BindingDesc, BindingInput};
    use crate::vk::resource::index::*;
    use crate::vk::{context::VkContext, GpuResources};

    // i'll likely replace this with a custom error type; this alias
    // makes that easier in the future
    pub type Result<T> = anyhow::Result<T>;

    pub type ResolverFn = Box<
        dyn FnOnce(&VkContext, &mut GpuResources, &mut Allocator) -> Result<()>
            + Send
            + Sync,
    >;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[repr(u64)]
    pub enum ResolveOrder {
        Buffer = 0,
        Image = 1,
        ImageView = 2,
        Sampler = 3,
        SampledImage = 4,
        BindingInput = 5,
        DescriptorSet = 6,
        Other = 7,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Priority {
        order: ResolveOrder,
        secondary: bool,
    }

    impl Priority {
        pub fn as_index(&self) -> usize {
            let a = self.order as usize;
            let b = self.secondary as usize;
            (a << 1) | b
        }

        pub fn primary(order: ResolveOrder) -> Self {
            Self {
                order,
                secondary: false,
            }
        }

        pub fn secondary(order: ResolveOrder) -> Self {
            Self {
                order,
                secondary: true,
            }
        }

        pub fn to_primary(self) -> Self {
            Self {
                secondary: false,
                ..self
            }
        }

        pub fn to_secondary(self) -> Self {
            Self {
                secondary: true,
                ..self
            }
        }

        pub fn is_primary(&self) -> bool {
            !self.secondary
        }

        pub fn is_secondary(&self) -> bool {
            self.secondary
        }
    }

    #[derive(Debug, Clone)]
    pub struct Resolvable<T: Copy> {
        value: Arc<AtomicCell<Option<T>>>,
        priority: Priority,
    }

    impl<T: Copy> Resolvable<T> {
        pub fn get(&self) -> Option<T> {
            self.value.load()
        }

        pub fn get_unwrap(&self) -> T {
            self.value.load().unwrap()
        }
    }

    #[derive(Clone)]
    pub struct BindableVar {
        ty: TypeId,
        value: Arc<AtomicCell<Option<rhai::Dynamic>>>,
    }

    #[derive(Default)]
    pub struct FrameBuilder {
        resolvers: BTreeMap<Priority, Vec<ResolverFn>>,
        variables: HashMap<String, BindableVar>,
        // variables: HashMap
        ast: rhai::AST,
        module: rhai::Module,
    }

    impl FrameBuilder {
        pub fn from_script(path: &str) -> anyhow::Result<Self> {
            use anyhow::anyhow;

            let builder = {
                let result = Self::default();
                let result = Arc::new(Mutex::new(result));

                let engine = Self::create_engine(result.clone());

                let path = std::path::PathBuf::from(path);
                let ast = engine.compile_file(path)?;

                let module = rhai::Module::eval_ast_as_new(
                    rhai::Scope::new(),
                    &ast,
                    &engine,
                )?;

                std::mem::drop(engine);

                let mutex = Arc::try_unwrap(result).map_err(|_| {
                    anyhow!(
                        "More than one Arc owner in FrameBuilder::from_script"
                    )
                })?;
                let mut result = mutex.into_inner();

                result.ast = ast;
                result.module = module;

                result
            };

            for (name, var) in builder.module.iter_var() {
                log::warn!("{} - {:?}", name, var);
            }

            Ok(builder)
        }

        fn create_engine(builder: Arc<Mutex<Self>>) -> rhai::Engine {
            let mut engine = super::create_batch_engine();

            engine.register_type_with_name::<BindableVar>("BindableVar");

            engine.register_fn("get", |var: BindableVar| {
                let v = var.value.take().unwrap();
                var.value.store(Some(v.clone()));
                v

                /*
                if var.ty == TypeId::of::<ImageIx>() {
                    let v = var.value.take().unwrap();
                    var.value.store(Some(v.clone()));
                    let i: ImageIx = v.cast();
                    rhai::Dynamic::from(i)
                } else if var.ty == TypeId::of::<ImageViewIx>() {
                    let v = var.value.take().unwrap();
                    var.value.store(Some(v.clone()));
                    let i: ImageViewIx = v.cast();
                    rhai::Dynamic::from(i)
                } else if var.ty == TypeId::of::<BufferIx>() {
                    let v = var.value.take().unwrap();
                    var.value.store(Some(v.clone()));
                    let i: BufferIx = v.cast();
                    rhai::Dynamic::from(i)
                } else if var.ty == TypeId::of::<PipelineIx>() {
                    let v = var.value.take().unwrap();
                    var.value.store(Some(v.clone()));
                    let i: PipelineIx = v.cast();
                    rhai::Dynamic::from(i)
                } else if var.ty == TypeId::of::<DescSetIx>() {
                    let v = var.value.take().unwrap();
                    var.value.store(Some(v.clone()));
                    let i: DescSetIx = v.cast();
                    rhai::Dynamic::from(i)
                } else {
                    let v = var.value.take().unwrap();
                    var.value.store(Some(v.clone()));
                    v
                }
                */
            });

            let b = builder.clone();
            engine.register_fn(
                "allocate_image",
                move |name: &str,
                      width: i64,
                      height: i64,
                      format: ash::vk::Format,
                      usage: ash::vk::ImageUsageFlags| {
                    b.lock().allocate_image(
                        width as u32,
                        height as u32,
                        format,
                        usage,
                        Some(name),
                    )
                },
            );

            let b = builder.clone();
            engine.register_fn(
                "image_view_for",
                move |image: Resolvable<ImageIx>| {
                    b.lock().create_image_view(image)
                },
            );

            let b = builder.clone();
            engine.register_fn(
                "load_compute_shader",
                move |path: &str, bindings: rhai::Array, pc_size: i64| {
                    let bindings = bindings
                        .into_iter()
                        .map(|b| b.cast::<BindingDesc>())
                        .collect::<Vec<_>>();

                    b.lock().load_compute_shader(
                        path,
                        &bindings,
                        pc_size as usize,
                    )
                },
            );

            let b = builder.clone();
            engine.register_fn("buffer_var", move |name: &str| {
                b.lock().new_var::<BufferIx>(name)
            });

            let b = builder.clone();
            engine.register_fn("image_var", move |name: &str| {
                b.lock().new_var::<ImageIx>(name)
            });

            let b = builder.clone();
            engine.register_fn("image_view_var", move |name: &str| {
                b.lock().new_var::<ImageViewIx>(name)
            });

            let b = builder.clone();
            engine.register_fn("pipeline_var", move |name: &str| {
                b.lock().new_var::<PipelineIx>(name)
            });

            let b = builder.clone();
            engine.register_fn("desc_set_var", move |name: &str| {
                b.lock().new_var::<DescSetIx>(name)
            });

            engine
        }

        pub fn new_var<T>(&mut self, k: &str) -> BindableVar
        where
            T: std::any::Any + Clone + Send + Sync,
        {
            let ty = TypeId::of::<T>();
            let value = Arc::new(AtomicCell::new(None));

            let var = BindableVar { ty, value };
            self.variables.insert(k.to_string(), var.clone());

            var
        }

        pub fn bind_var<T>(&mut self, k: &str, v: T) -> anyhow::Result<()>
        where
            T: std::any::Any + Clone + Send + Sync,
        {
            use anyhow::{anyhow, bail};

            let var = self.variables.get(k).ok_or(anyhow!(
                "attempted to bind nonexistent variable '{}'",
                k
            ))?;

            let expected_ty = TypeId::of::<T>();

            if var.ty != expected_ty {
                bail!("attempted to bind value of incorrect type to variable '{}'", k);
            }

            var.value.store(Some(rhai::Dynamic::from(v)));

            Ok(())
        }

        pub fn allocate_buffer(
            &mut self,
            size: u64,
            usage: ash::vk::BufferUsageFlags,
            name: Option<&str>,
        ) -> Resolvable<ImageIx> {
            todo!();
        }

        pub fn allocate_image(
            &mut self,
            width: u32,
            height: u32,
            format: ash::vk::Format,
            usage: ash::vk::ImageUsageFlags,
            name: Option<&str>,
        ) -> Resolvable<ImageIx> {
            let priority = Priority::primary(ResolveOrder::Image);

            let cell = Arc::new(AtomicCell::new(None));
            let name = name.map(String::from);
            let inner = cell.clone();

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      alloc: &mut Allocator| {
                    let img = res.allocate_image(
                        ctx,
                        alloc,
                        width,
                        height,
                        format,
                        usage,
                        name.as_ref().map(|s| s.as_str()),
                    )?;
                    inner.store(Some(img));
                    Ok(())
                },
            ) as ResolverFn;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn create_image_view(
            &mut self,
            image_ix: Resolvable<ImageIx>,
            // name: Option<&str>,
        ) -> Resolvable<ImageViewIx> {
            let priority = Priority::primary(ResolveOrder::ImageView);

            let cell = Arc::new(AtomicCell::new(None));
            // let name = name.map(String::from);
            let inner = cell.clone();

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      _: &mut Allocator| {
                    let image = image_ix.value.load().unwrap();
                    let view = res.create_image_view_for_image(ctx, image)?;
                    inner.store(Some(view));
                    Ok(())
                },
            ) as ResolverFn;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn create_desc_set(
            &mut self,
            stage_flags: ash::vk::ShaderStageFlags,
            binding_descs: &[BindingDesc],
            dyn_inputs: rhai::Array,
        ) -> Resolvable<DescSetIx> {
            let mut inputs: Vec<BindingInput> = Vec::new();

            for v in dyn_inputs {
                let map: rhai::Map = v.cast();

                log::warn!("extracting type");
                let ty = map.get("type").unwrap();
                let ty_str = ty.clone().into_string().unwrap();

                log::warn!("extracting binding");
                let binding = map.get("binding").unwrap().clone();
                let binding = binding.as_int().unwrap() as u32;

                match ty_str.as_str() {
                    "image_view" => {
                        log::warn!("extracting view");
                        let view = map.get("view").unwrap().clone();
                        let view = view.cast();
                        let input = BindingInput::ImageView { binding, view };
                        inputs.push(input);
                    }
                    "buffer" => {
                        log::warn!("extracting buffer");
                        let buffer = map.get("buffer").unwrap().clone();
                        let buffer = buffer.cast();
                        let input = BindingInput::Buffer { binding, buffer };
                        inputs.push(input);
                    }
                    _ => panic!("unsupported binding input type"),
                }
            }

            let cell = Arc::new(AtomicCell::new(None));
            // let name = name.map(String::from);
            let inner = cell.clone();

            let binding_descs = binding_descs.to_owned();

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      alloc: &mut Allocator| {
                    let desc_set = res.allocate_desc_set(
                        &binding_descs,
                        &inputs,
                        stage_flags,
                    )?;
                    inner.store(Some(desc_set));
                    Ok(())
                },
            ) as ResolverFn;

            let priority = Priority::primary(ResolveOrder::DescriptorSet);

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn load_compute_shader(
            &mut self,
            shader_path: &str,
            bindings: &[BindingDesc],
            pc_size: usize,
        ) -> Resolvable<PipelineIx> {
            let priority = Priority::primary(ResolveOrder::Other);

            let cell = Arc::new(AtomicCell::new(None));
            // let name = name.map(String::from);
            let inner = cell.clone();

            let shader_path = shader_path.to_string();
            let bindings = Vec::from(bindings);

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      _: &mut Allocator| {
                    let pipeline = res.load_compute_shader_runtime(
                        ctx,
                        &shader_path,
                        &bindings,
                        pc_size,
                    )?;
                    inner.store(Some(pipeline));
                    Ok(())
                },
            ) as ResolverFn;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn resolve_and_then<
            T: Copy + Send + Sync + 'static,
            U: Copy + Send + Sync + 'static,
        >(
            &mut self,
            f: impl FnOnce(T) -> U + Send + Sync + 'static,
            r: Resolvable<T>,
        ) -> Resolvable<U> {
            // let priority = r.priority.as
            let priority = r.priority.to_secondary();

            let cell = Arc::new(AtomicCell::new(None));
            let inner = cell.clone();

            let resolver = Box::new(
                move |_: &VkContext,
                      _: &mut GpuResources,
                      _: &mut Allocator| {
                    let t = r.get_unwrap();
                    let u = f(t);
                    inner.store(Some(u));
                    Ok(())
                },
            ) as ResolverFn;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn add_resolvable<F, T>(
            &mut self,
            priority: Priority,
            f: F,
        ) -> Resolvable<T>
        where
            T: Copy + Send + Sync + 'static,
            F: FnOnce(
                    &VkContext,
                    &mut GpuResources,
                    &mut Allocator,
                ) -> Result<T>
                + Send
                + Sync
                + 'static,
        {
            let cell = Arc::new(AtomicCell::new(None));
            let inner = cell.clone();

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      alloc: &mut Allocator| {
                    let val = f(ctx, res, alloc)?;
                    inner.store(Some(val));
                    Ok(())
                },
            ) as ResolverFn;

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }
    }
}
