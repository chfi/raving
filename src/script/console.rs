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
    // images: FxHashMap<usize, Resolvable<ImageIx>>,
    rhai_mod: rhai::Module,
}

/// the minimal engine
pub fn create_engine() -> rhai::Engine {
    let mut engine = rhai::Engine::new();

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

    engine
}

impl ModuleBuilder {
    pub fn from_script(path: &str) -> anyhow::Result<Self> {
        let mut engine = create_engine();

        let ast = {
            let result = Self::default();
            let arcres = Arc::new(Mutex::new(result));

            let res = arcres.clone();

            engine.register_fn(
                "allocate_image",
                move |width: u32,
                      height: u32,
                      format: vk::Format,
                      usage: vk::ImageUsageFlags| {
                    let resolvable =
                        res.lock().allocate_image(width, height, format, usage);
                    resolvable
                },
            );

            let res = arcres.clone();

            // engine.register_fn(
            //     "image_var"
            //         move |name: &str| {

            //         });

            let path = std::path::PathBuf::from(path);
            let ast = engine.compile_file(path)?;

            ast
        };

        let module =
            rhai::Module::eval_ast_as_new(rhai::Scope::new(), &ast, &engine)?;

        // module.build_index()

        for (name, var) in module.iter_var() {
            log::warn!("{} - {:?}", name, var);
        }

        // result.rhai_mod = module;

        // Ok(result)
        todo!();
    }

    pub fn bind_image_var(&mut self, k: &str, v: ImageIx) -> Option<()> {
        let var = self.image_vars.remove(k)?;
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_image_view_var(
        &mut self,
        k: &str,
        v: ImageViewIx,
    ) -> Option<()> {
        let var = self.image_view_vars.remove(k)?;
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_buffer_var(&mut self, k: &str, v: BufferIx) -> Option<()> {
        let var = self.buffer_vars.remove(k)?;
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_pipeline_var(&mut self, k: &str, v: PipelineIx) -> Option<()> {
        let var = self.pipeline_vars.remove(k)?;
        var.value.store(Some(v));
        Some(())
    }

    pub fn bind_desc_set_var(&mut self, k: &str, v: DescSetIx) -> Option<()> {
        let var = self.desc_set_vars.remove(k)?;
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

#[derive(Clone, Default)]
pub struct Resolvable<T: Copy> {
    value: Arc<AtomicCell<Option<T>>>,
}

impl<T: Copy> Resolvable<T> {
    pub fn new(value: Arc<AtomicCell<Option<T>>>) -> Self {
        Self { value }
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

pub type AllocResolver = Box<
    dyn FnOnce(&VkContext, &mut GpuResources, &mut Allocator) + Send + Sync,
>;

pub type GpuCmd = Arc<
    dyn Fn(&ash::Device, &GpuResources, vk::CommandBuffer)
        + Send
        + Sync
        + 'static,
>;

// pub struct VariableMap<T> {
//     map: FxHashMap<usize
// }

pub enum Resolver<T> {
    Resolver(Option<WithAllocators<T>>),
    Value(T),
}

impl<T> Resolver<T> {
    pub fn resolve(
        &mut self,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> Option<&T> {
        match self {
            Resolver::Resolver(resolver) => {
                let f = resolver.take()?;

                let v = f(ctx, res, alloc).ok()?;

                *self = Self::Value(v);

                self.get()
            }
            Resolver::Value(v) => Some(v),
        }
    }

    pub fn get(&self) -> Option<&T> {
        if let Self::Value(v) = &self {
            Some(v)
        } else {
            None
        }
    }

    pub fn can_resolve(&self) -> bool {
        !matches!(self, Self::Resolver(None))
    }

    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Value(_))
    }
}

impl<T> std::default::Default for Resolver<T> {
    fn default() -> Self {
        Self::Resolver(None)
    }
}

pub type VariableMap<T> = Arc<Mutex<FxHashMap<usize, Mutex<Resolver<T>>>>>;
// Arc<Mutex<FxHashMap<usize, Mutex<Option<WithAllocators<T>>>>>>;

#[derive(Default, Clone)]
pub struct BatchBuilder {
    image_vars: VariableMap<ImageIx>,
    image_view_vars: Arc<Mutex<FxHashMap<usize, usize>>>,

    buffer_vars: VariableMap<BufferIx>,

    pipeline_vars: VariableMap<PipelineIx>,

    desc_set_vars: VariableMap<DescSetIx>,
    // cmds: Arc<Mutex<FxHashMap<usize, GpuCmd>>>,
}

impl BatchBuilder {
    pub fn allocate_image(
        &self,
        w: u32,
        h: u32,
        fmt: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> usize {
        let resolver = Box::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                res.allocate_image(ctx, alloc, w, h, fmt, usage, None)
            },
        ) as WithAllocators<ImageIx>;
        self.new_image_resolver(resolver)
    }

    pub fn new_image_resolver(
        &self,
        resolver: WithAllocators<ImageIx>,
    ) -> usize {
        let mut vars = self.image_vars.lock();

        let i = vars.len();

        vars.insert(i, Mutex::new(Resolver::Resolver(Some(resolver))));

        i
    }

    pub fn new_image_view(&self, img: usize) -> Option<usize> {
        let mut vars = self.image_view_vars.lock();
        let i = vars.len();
        vars.insert(i, img);
        Some(i)
    }

    pub fn get_image(&self, img: usize) -> Option<ImageIx> {
        let mut vars = self.image_vars.lock();
        let resolver = vars.get_mut(&img)?;
        let r = resolver.get_mut();
        let v = r.get()?;
        Some(*v)
    }

    pub fn resolve_image(
        &self,
        img: usize,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> Option<ImageIx> {
        let mut vars = self.image_vars.lock();
        let resolver = vars.get_mut(&img)?;
        let r = resolver.get_mut();
        let v = r.resolve(ctx, res, alloc)?;
        Some(*v)
    }

    pub fn create_engine(&self) -> rhai::Engine {
        let mut engine = rhai::Engine::new();

        let self_arc = Arc::new(self.clone());
        let other = self_arc.clone();

        engine.register_result_fn(
            "allocate_image",
            move |width: i64,
                  height: i64,
                  fmt: ash::vk::Format,
                  usage: ash::vk::ImageUsageFlags| {
                let img = other.allocate_image(
                    width as u32,
                    height as u32,
                    fmt,
                    usage,
                );

                Ok(img)
            },
        );

        let other = self_arc.clone();

        engine.register_result_fn(
            "create_image_view",
            move |image_var: usize| {
                let i = other.new_image_view(image_var).unwrap();
                Ok(i)
            },
        );

        todo!();
    }
}

pub struct Console {
    // input_history: Vec<String>,
    output_log: Vec<String>,

    renderer: LineRenderer,
}

impl Console {
    pub fn create_engine() -> rhai::Engine {
        let mut engine = rhai::Engine::new();

        todo!();
    }
}
