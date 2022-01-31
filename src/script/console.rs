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
    init_fn: Vec<InitFn>,

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

/// the minimal engine
pub fn create_engine() -> rhai::Engine {
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
            let mut engine = create_engine();

            let result = Self::default();
            let arcres = Arc::new(Mutex::new(result));

            let res = arcres.clone();
            engine.register_fn(
                "allocate_image",
                move |width: i64,
                      height: i64,
                      format: vk::Format,
                      usage: vk::ImageUsageFlags| {
                    let resolvable = res.lock().allocate_image(
                        width as u32,
                        height as u32,
                        format,
                        usage,
                    );
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

    pub fn allocate_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Resolvable<ImageIx> {
        let resolvable = Arc::new(AtomicCell::new(None));

        let inner = resolvable.clone();

        let resolver = Box::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                let img = res.allocate_image(
                    ctx, alloc, width, height, format, usage, None,
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

#[derive(Clone)]
pub struct Resolvable<T: Copy> {
    value: Arc<AtomicCell<Option<T>>>,
}

impl<T: Copy> Resolvable<T> {
    pub fn new(value: Arc<AtomicCell<Option<T>>>) -> Self {
        Self { value }
    }

    pub fn empty() -> Self {
        Self {
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
