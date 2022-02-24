use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::any::TypeId;
use std::collections::HashMap;
use std::{collections::BTreeMap, sync::Arc};

use crate::vk::descriptor::DescriptorUpdateBuilder;
use crate::vk::resource::index::*;
use crate::vk::{context::VkContext, GpuResources};
use crate::vk::{FrameResources, ImageRes};

use super::WithAllocatorsInput;

// i'll likely replace this with a custom error type; this alias
// makes that easier in the future
pub type Result<T> = anyhow::Result<T>;

pub type ResolverFn = Box<
    dyn FnOnce(&VkContext, &mut GpuResources, &mut Allocator) -> Result<()>
        + Send
        + Sync,
>;

#[derive(Debug, Clone)]
pub struct Resolvable<T: Copy> {
    pub(super) value: Arc<AtomicCell<Option<T>>>,
    pub(super) priority: Priority,
}

pub struct RhaiFrame {
    resources: FrameResources,
    builder: FrameBuilder,
}

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

pub fn try_get_var<T: Copy + std::any::Any + 'static>(
    val: &rhai::Dynamic,
) -> Option<T> {
    let ty = val.type_id();

    if ty == TypeId::of::<BindableVar>() {
        let var: BindableVar = val.clone_cast();
        if var.ty == TypeId::of::<T>() {
            let val = var.value.take();
            var.value.store(val.clone());
            return val.and_then(|v| v.try_cast());
        } else {
            return None;
        }
    }

    if ty == TypeId::of::<Resolvable<T>>() {
        let res: Resolvable<T> = val.clone_cast();
        return res.get();
    }

    None
}

#[derive(Default)]
pub struct FrameBuilder {
    resolvers: BTreeMap<Priority, Vec<ResolverFn>>,
    variables: HashMap<String, BindableVar>,
    // variables: HashMap
    pub ast: rhai::AST,
    pub module: rhai::Module,
    /*
    swapchain_dependent_images:
        FxHashMap<ImageIx, WithAllocatorsInput<[u32; 2], ImageRes>>,
    swapchain_dependent_image_views: FxHashMap<
        ImageViewIx,
        WithAllocatorsInput<ImageIx, ash::vk::ImageView>,
    >,

    // not sure about the input type here
    swapchain_dependent_desc_sets:
        FxHashMap<DescSetIx, WithAllocatorsInput<(), ash::vk::DescriptorSet>>,
    */
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
        engine.register_fn("create_sampler", move |builder: rhai::Map| {
            b.lock().add_resolvable(
                Priority::primary(ResolveOrder::Sampler),
                move |ctx, res, _alloc| {
                    let sampler_info =
                        crate::script::vk::sampler_create_info::build(builder)?;
                    res.insert_sampler(ctx, sampler_info)
                },
            )
        });

        let b = builder.clone();
        engine.register_fn(
            "load_shader",
            move |shader_path: &str, stage: ash::vk::ShaderStageFlags| {
                b.lock().load_shader(shader_path, stage)
            },
        );

        let b = builder.clone();
        engine.register_fn(
            "create_compute_pipeline",
            move |shader: Resolvable<ShaderIx>| {
                b.lock().create_compute_pipeline(shader)
            },
        );

        let b = builder.clone();
        engine.register_fn(
            "create_desc_set",
            move |shader: Resolvable<ShaderIx>,
                  set: i64,
                  inputs: rhai::Array| {
                let inputs = inputs
                    .into_iter()
                    .map(|map| map.cast::<rhai::Map>())
                    .collect::<Vec<_>>();

                b.lock().create_desc_set(shader, set as u32, inputs)
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
                let buf = res.allocate_buffer(
                    ctx,
                    alloc,
                    location,
                    elem_size as usize,
                    size as usize,
                    usage,
                    name.as_ref().map(|s| s.as_str()),
                )?;
                Ok(res.insert_buffer(buf))
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
                let img = res.allocate_image(
                    ctx,
                    alloc,
                    width,
                    height,
                    format,
                    usage,
                    name.as_ref().map(|s| s.as_str()),
                )?;
                Ok(res.insert_image(img))
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
                let image = &res[image];
                let image_view = res.new_image_view(ctx, image)?;
                Ok(res.insert_image_view(image_view))
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
            move |_ctx, res, _alloc| {
                let shader = res.load_shader(&shader_path, stage_flags)?;
                Ok(res.insert_shader(shader))
            },
        )
    }

    pub fn create_compute_pipeline(
        &mut self,
        shader: Resolvable<ShaderIx>,
    ) -> Resolvable<PipelineIx> {
        self.add_resolvable(
            Priority::primary(ResolveOrder::Pipeline),
            move |ctx, res, _alloc| {
                let shader = shader.get().unwrap();
                res.create_compute_pipeline(ctx, shader)
            },
        )
    }

    pub fn create_desc_set(
        &mut self,
        shader: Resolvable<ShaderIx>,
        set: u32,
        inputs: Vec<rhai::Map>,
    ) -> Resolvable<DescSetIx> {
        let priority = Priority::primary(ResolveOrder::DescriptorSet);
        use anyhow::anyhow;

        fn append_input(
            res: &GpuResources,
            builder: &mut DescriptorUpdateBuilder,
            input: &rhai::Map,
        ) -> Result<()> {
            use ash::vk::DescriptorType as Ty;

            let binding = input
                .get("binding")
                .and_then(|b| b.as_int().ok())
                .ok_or(anyhow!("input lacks binding field"))?;

            let binding = binding as u32;

            let ty = builder
                .binding_desc_type(binding)
                .ok_or(anyhow!("descriptor set lacks binding {}", binding))?;

            if let Some(buf_ix) =
                input.get("buffer").and_then(|b| try_get_var::<BufferIx>(b))
            {
                let buffer = &res[buf_ix];
                let buf_info = ash::vk::DescriptorBufferInfo::builder()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(ash::vk::WHOLE_SIZE)
                    .build();

                let buffer_info = [buf_info];
                builder.bind_buffer(binding, &buffer_info);
            }

            // let mut img_info = None;

            if let Some(sampler_ix) = input
                .get("sampler")
                .and_then(|b| try_get_var::<SamplerIx>(b))
            {
                log::warn!("adding sampler");
                let sampler = res[sampler_ix];

                let info = ash::vk::DescriptorImageInfo::builder()
                    .sampler(sampler)
                    .build();

                builder.bind_image(binding, &[info]);
            }

            if let Some(img_view_ix) = input
                .get("image_view")
                .and_then(|b| try_get_var::<ImageViewIx>(b))
            {
                let layout = if ty == Ty::STORAGE_IMAGE {
                    ash::vk::ImageLayout::GENERAL
                } else {
                    ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                };

                let view = res[img_view_ix];

                let info = ash::vk::DescriptorImageInfo::builder()
                    .image_layout(layout)
                    .image_view(view)
                    .build();

                builder.bind_image(binding, &[info]);
            }

            // if let Some(img_info) = img_info {
            //     let info = img_info.build();
            //     builder.bind_image(binding, &[info]);
            // }

            Ok(())
        }

        self.add_resolvable(
            priority,
            move |_ctx: &VkContext,
                  res: &mut GpuResources,
                  _alloc: &mut Allocator| {
                let shader = shader.value.load().unwrap();
                let desc_set =
                    res.allocate_desc_set(shader, set, |res, builder| {
                        for input in inputs {
                            append_input(res, builder, &input)?;
                        }

                        Ok(())
                    })?;
                Ok(res.insert_desc_set(desc_set))
            },
        )
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
