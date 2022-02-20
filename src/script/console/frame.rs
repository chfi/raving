use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::any::TypeId;
use std::collections::HashMap;
use std::{collections::BTreeMap, sync::Arc};

use crate::vk::descriptor::BindingDesc;
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
    Shader = 0,
    Pipeline = 1,
    Buffer = 2,
    Image = 3,
    ImageView = 4,
    Sampler = 5,
    SampledImage = 6,
    BindingInput = 7,
    DescriptorSet = 8,
    Other = 9,
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
    pub(super) value: Arc<AtomicCell<Option<T>>>,
    pub(super) priority: Priority,
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
    pub(crate) ty: TypeId,
    pub(crate) value: Arc<AtomicCell<Option<rhai::Dynamic>>>,
}

#[derive(Default)]
pub struct FrameBuilder {
    resolvers: BTreeMap<Priority, Vec<ResolverFn>>,
    variables: HashMap<String, BindableVar>,
    // variables: HashMap
    pub ast: rhai::AST,
    pub module: rhai::Module,
}

impl FrameBuilder {
    pub fn resolve(
        &mut self,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> anyhow::Result<bool> {
        // TODO check that all variables have been bound

        for (priority, resolvers) in self.resolvers.iter_mut() {
            log::warn!("{:?} - {} resolvers", priority, resolvers.len());

            for f in resolvers.drain(..) {
                f(ctx, res, alloc)?;
            }
        }

        Ok(self.is_resolved())
    }

    pub fn is_resolved(&self) -> bool {
        let no_resolvers = self.resolvers.values().all(|v| v.is_empty());

        // let no_unbound = self.variables.values()
        no_resolvers
    }

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
                anyhow!("More than one Arc owner in FrameBuilder::from_script")
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
        });

        let b = builder.clone();
        engine.register_fn(
            "allocate_buffer",
            move |name: &str,
                  location: MemoryLocation,
                  elem_size: i64,
                  elems: i64,
                  usage: ash::vk::BufferUsageFlags| {
                b.lock().allocate_buffer(
                    elem_size,
                    elems,
                    location,
                    usage,
                    Some(name),
                )
            },
        );

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
            move |image: Resolvable<ImageIx>| b.lock().create_image_view(image),
        );

        engine.register_result_fn(
            "read_shader_descriptors",
            move |shader_path: &str| {
                let comp_src = {
                    let mut file = std::fs::File::open(shader_path).unwrap();
                    ash::util::read_spv(&mut file).unwrap()
                };

                let (sets, pcs) = rspirv_reflect::Reflection::new_from_spirv(
                    bytemuck::cast_slice(&comp_src),
                )
                .and_then(|i| {
                    let sets = i.get_descriptor_sets().unwrap();
                    let pcs = i.get_push_constant_range().unwrap().unwrap();
                    Ok((sets, pcs))
                })
                .unwrap();

                let mut result = rhai::Map::default();

                for (set, contents) in sets {
                    let contents = rhai::Dynamic::from(contents);
                    result.insert(set.to_string().into(), contents);
                }

                result.insert(
                    "push_constant_offset".into(),
                    rhai::Dynamic::from(pcs.offset),
                );
                result.insert(
                    "push_constant_size".into(),
                    rhai::Dynamic::from(pcs.size),
                );

                Ok(result)
            },
        );

        let b = builder.clone();
        engine.register_fn(
            "load_shader",
            move |shader_path: &str, stage: ash::vk::ShaderStageFlags| {
                b.lock().load_shader(shader_path, stage)
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

                b.lock()
                    .load_compute_shader(path, &bindings, pc_size as usize)
            },
        );

        let b = builder.clone();
        engine.register_fn(
                "create_desc_set",
                move |stage_flags: ash::vk::ShaderStageFlags,
                      set_infos: rhai::Map,
                      set: i64,
                      inputs: rhai::Array| {
                    let key = set.to_string();

                    let set_info = set_infos.get(key.as_str()).unwrap();
                    let set_info = set_info
                              .clone_cast::<BTreeMap<u32, rspirv_reflect::DescriptorInfo>>();

                    let inputs = inputs
                        .into_iter()
                        .map(|map| map.cast::<rhai::Map>())
                        .collect::<Vec<_>>();

                    b.lock().create_desc_set(stage_flags, set_info, inputs)
                },
            );

        /*
        let b = builder.clone();
        engine.register_fn(
            "create_desc_set",
            move |stage_flags: ash::vk::ShaderStageFlags,
                  binding_descs: rhai::Array,
                  inputs: rhai::Array| {
                let bindings: Vec<BindingDesc> =
                    binding_descs.into_iter().map(|b| b.cast()).collect();

                b.lock().create_desc_set(stage_flags, &bindings, inputs)
            },
        );
        */

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

        let var = self
            .variables
            .get(k)
            .ok_or(anyhow!("attempted to bind nonexistent variable '{}'", k))?;

        let expected_ty = TypeId::of::<T>();

        if var.ty != expected_ty {
            bail!(
                "attempted to bind value of incorrect type to variable '{}'",
                k
            );
        }

        var.value.store(Some(rhai::Dynamic::from(v)));

        Ok(())
    }

    pub fn allocate_buffer(
        &mut self,
        elem_size: i64,
        size: i64,
        location: gpu_allocator::MemoryLocation,
        usage: ash::vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> Resolvable<BufferIx> {
        let name = name.map(String::from);
        self.add_resolvable(
            Priority::primary(ResolveOrder::Buffer),
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                res.allocate_buffer(
                    ctx,
                    alloc,
                    location,
                    elem_size as usize,
                    size as usize,
                    usage,
                    name.as_ref().map(|s| s.as_str()),
                )
            },
        )
    }

    pub fn allocate_image(
        &mut self,
        width: u32,
        height: u32,
        format: ash::vk::Format,
        usage: ash::vk::ImageUsageFlags,
        name: Option<&str>,
    ) -> Resolvable<ImageIx> {
        let name = name.map(String::from);
        self.add_resolvable(
            Priority::primary(ResolveOrder::Image),
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                res.allocate_image(
                    ctx,
                    alloc,
                    width,
                    height,
                    format,
                    usage,
                    name.as_ref().map(|s| s.as_str()),
                )
            },
        )
    }

    pub fn create_image_view(
        &mut self,
        image_ix: Resolvable<ImageIx>,
        // name: Option<&str>,
    ) -> Resolvable<ImageViewIx> {
        let priority = Priority::primary(ResolveOrder::ImageView);

        self.add_resolvable(
            priority,
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  _: &mut Allocator| {
                let image = image_ix.value.load().unwrap();
                res.create_image_view_for_image(ctx, image)
            },
        )
    }

    pub fn load_shader(
        &mut self,
        path: &str,
        stage_flags: ash::vk::ShaderStageFlags,
    ) -> Resolvable<ShaderIx> {
        let shader_path = path.to_string();
        self.add_resolvable(
            Priority::primary(ResolveOrder::Shader),
            move |_ctx, res, _alloc| res.load_shader(&shader_path, stage_flags),
        )
    }

    pub fn create_compute_pipeline(
        &mut self,
        shader: ShaderIx,
    ) -> Resolvable<PipelineIx> {
        self.add_resolvable(
            Priority::primary(ResolveOrder::Pipeline),
            move |ctx, res, _alloc| res.create_compute_pipeline(ctx, shader),
        )
    }

    /*
    pub fn create_desc_set_(
        &mut self,
        shader: ShaderIx,
        set: u32,
        inputs: Vec<rhai::Map>,
    ) -> Resolvable<DescSetIx> {
        let priority = Priority::primary(ResolveOrder::DescriptorSet);
        use anyhow::anyhow;

        #[rustfmt::skip]
    macro_rules! get_and_cast {
        ($map:ident, $field:literal, $type:ty) => {
            {
                let val = $map.get($field).unwrap();

                if val.type_id() == TypeId::of::<BindableVar>() {
                    let var = val.clone_cast::<BindableVar>();
                    let bufdyn = var.value.take();
                    var.value.store(bufdyn.clone());
                    let res =
                        bufdyn.unwrap().cast::<$type>();
                    res
                } else {
                    let res = val.clone_cast::<Resolvable<$type>>();
                    res.get_unwrap()
                }
            }
        };
    }

        self.add_resolvable(
            priority,
            move |_ctx: &VkContext,
                  res: &mut GpuResources,
                  _alloc: &mut Allocator| {
                res.allocate_desc_set(shader, set, |builder| {
                    for input in inputs {

                    }

                    Ok(())
                })

                // res.allocate_desc_set_dyn(stage_flags, &set_info, &inputs)
            },
        )
    }
    */

    pub fn create_desc_set(
        &mut self,
        stage_flags: ash::vk::ShaderStageFlags,
        set_info: BTreeMap<u32, rspirv_reflect::DescriptorInfo>,
        inputs: Vec<rhai::Map>,
    ) -> Resolvable<DescSetIx> {
        let priority = Priority::primary(ResolveOrder::DescriptorSet);
        self.add_resolvable(
            priority,
            move |_ctx: &VkContext,
                  res: &mut GpuResources,
                  _alloc: &mut Allocator| {
                res.allocate_desc_set_dyn(stage_flags, &set_info, &inputs)
            },
        )
    }

    /*
    pub fn create_desc_set_old(
        &mut self,
        stage_flags: ash::vk::ShaderStageFlags,
        binding_descs: &[BindingDesc],
        dyn_inputs: rhai::Array,
    ) -> Resolvable<DescSetIx> {
        let cell = Arc::new(AtomicCell::new(None));
        // let name = name.map(String::from);
        let inner = cell.clone();

        let binding_descs = binding_descs.to_owned();

        let resolver = Box::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                let mut inputs: Vec<BindingInput> = Vec::new();

                // TODO this has to be cleaned up a ton
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

                            let input = if let Some(view) =
                                view.clone()
                                    .try_cast::<Resolvable<ImageViewIx>>()
                            {
                                let view = view.get_unwrap();
                                let input = BindingInput::ImageView {
                                    binding,
                                    view,
                                };
                                input
                            } else if let Some(var) =
                                view.clone().try_cast::<BindableVar>()
                            {
                                let bufdyn = var.value.take();
                                var.value.store(bufdyn.clone());

                                let view =
                                    bufdyn.unwrap().cast::<ImageViewIx>();

                                let input = BindingInput::ImageView {
                                    binding,
                                    view,
                                };
                                input
                            } else {
                                unreachable!();
                            };

                            inputs.push(input);
                        }
                        "buffer" => {
                            log::warn!("extracting buffer");
                            let buffer = map.get("buffer").unwrap().clone();

                            let input = if let Some(buffer) = buffer
                                .clone()
                                .try_cast::<Resolvable<BufferIx>>()
                            {
                                let buffer = buffer.get_unwrap();
                                let input = BindingInput::Buffer {
                                    binding,
                                    buffer,
                                };
                                input
                            } else if let Some(var) =
                                buffer.clone().try_cast::<BindableVar>()
                            {
                                let bufdyn = var.value.take();
                                var.value.store(bufdyn.clone());

                                let buffer =
                                    bufdyn.unwrap().cast::<BufferIx>();

                                let input = BindingInput::Buffer {
                                    binding,
                                    buffer,
                                };
                                input
                            } else {
                                unreachable!();
                            };

                            inputs.push(input);
                        }
                        _ => panic!("unsupported binding input type"),
                    }
                }

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
    */

    pub fn load_compute_shader(
        &mut self,
        shader_path: &str,
        bindings: &[BindingDesc],
        pc_size: usize,
    ) -> Resolvable<PipelineIx> {
        let priority = Priority::primary(ResolveOrder::Other);
        let shader_path = shader_path.to_string();
        let bindings = Vec::from(bindings);

        self.add_resolvable(
            priority,
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  _: &mut Allocator| {
                res.load_compute_shader_runtime(
                    ctx,
                    &shader_path,
                    &bindings,
                    pc_size,
                )
            },
        )
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
            move |_: &VkContext, _: &mut GpuResources, _: &mut Allocator| {
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
        F: FnOnce(&VkContext, &mut GpuResources, &mut Allocator) -> Result<T>
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
