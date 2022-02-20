use ash::vk;
use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;

use rhai::NativeCallContext;
use rspirv_reflect::Reflection;

use std::{any::TypeId, sync::Arc};

use crate::vk::{
    context::VkContext, resource::index::*, GpuResources, VkEngine,
};

use frame::*;

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

    pub fn initialize_buffer_with(
        &mut self,
        dst: BufferIx,
        data: Vec<u8>,
        // repeat: bool,
    ) {
        let staging_bufs = self.staging_buffers.clone();

        let f = Arc::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator,
                  cmd: vk::CommandBuffer| {
                let buffer = &mut res[dst];
                // let buf_size = buffer.size_bytes();

                let staging = buffer.upload_to_self_bytes(
                    ctx,
                    alloc,
                    data.as_slice(),
                    cmd,
                )?;

                let stg_ix = res.insert_buffer(staging);

                staging_bufs.lock().push(stg_ix);

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

pub fn create_batch_engine() -> rhai::Engine {
    let mut engine = rhai::Engine::new();

    // by default, debug builds are set to (32, 16), but 16 is too low
    // for some scripts in practice
    engine.set_max_expr_depths(64, 32);

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

    engine.register_fn("get", |var: BindableVar| {
        let v = var.value.take().unwrap();
        var.value.store(Some(v.clone()));
        v
    });

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

    engine.register_fn("atomic_int", |v: i64| Arc::new(AtomicCell::new(v)));
    engine.register_fn("atomic_float", |v: f32| Arc::new(AtomicCell::new(v)));
    engine.register_fn("get", |v: Arc<AtomicCell<i64>>| v.load());
    engine.register_fn("get", |v: Arc<AtomicCell<f32>>| v.load());
    engine
        .register_fn("set", |c: &mut Arc<AtomicCell<i64>>, v: i64| c.store(v));
    engine
        .register_fn("set", |c: &mut Arc<AtomicCell<f32>>, v: f32| c.store(v));

    engine.register_fn("append_str_tmp", |blob: &mut Vec<u8>, text: &str| {
        let mut write_u32 = |u: u32| {
            for b in u.to_le_bytes() {
                blob.push(b);
            }
        };

        for &b in text.as_bytes() {
            write_u32(b as u32);
        }
    });

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
        "initialize_buffer_with",
        |builder: &mut BatchBuilder, buffer: BufferIx, data: Vec<u8>| {
            builder.initialize_buffer_with(buffer, data);
        },
    );

    let arg_types = [
        TypeId::of::<BatchBuilder>(),
        TypeId::of::<rhai::FnPtr>(),
        TypeId::of::<BufferIx>(),
        TypeId::of::<Vec<rhai::Dynamic>>(),
    ];
    engine.register_raw_fn(
        "initialize_buffer_with",
        &arg_types,
        move |ctx: NativeCallContext, args: &mut [&'_ mut rhai::Dynamic]| {
            let fn_ptr: rhai::FnPtr = args.get(1).unwrap().clone_cast();
            let buffer: BufferIx = args.get(2).unwrap().clone_cast();
            let vals: rhai::Array = std::mem::take(args[3]).cast();

            let mut output: Vec<u8> = Vec::new();

            for val in vals {
                let entry: Vec<u8> =
                    fn_ptr.call_within_context(&ctx, (val,))?;
                output.extend(entry);
            }

            let mut builder = args[0].write_lock::<BatchBuilder>().unwrap();
            builder.initialize_buffer_with(buffer, output);

            Ok(())
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

pub type WithAllocators<T> = Box<
    dyn FnOnce(
            &VkContext,
            &mut GpuResources,
            &mut Allocator,
        ) -> anyhow::Result<T>
        + Send
        + Sync,
>;

pub mod frame {

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
                move |image: Resolvable<ImageIx>| {
                    b.lock().create_image_view(image)
                },
            );

            engine.register_result_fn(
                "read_shader_descriptors",
                move |shader_path: &str| {
                    let comp_src = {
                        let mut file =
                            std::fs::File::open(shader_path).unwrap();
                        ash::util::read_spv(&mut file).unwrap()
                    };

                    let (sets, pcs) =
                        rspirv_reflect::Reflection::new_from_spirv(
                            bytemuck::cast_slice(&comp_src),
                        )
                        .and_then(|i| {
                            let sets = i.get_descriptor_sets().unwrap();
                            let pcs =
                                i.get_push_constant_range().unwrap().unwrap();
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

                    b.lock().load_compute_shader(
                        path,
                        &bindings,
                        pc_size as usize,
                    )
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
            elem_size: i64,
            size: i64,
            location: gpu_allocator::MemoryLocation,
            usage: ash::vk::BufferUsageFlags,
            name: Option<&str>,
        ) -> Resolvable<BufferIx> {
            let priority = Priority::primary(ResolveOrder::Buffer);

            let cell = Arc::new(AtomicCell::new(None));
            let name = name.map(String::from);
            let inner = cell.clone();

            let resolver = Box::new(
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
                    inner.store(Some(buf));
                    Ok(())
                },
            ) as ResolverFn;

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

        pub fn load_shader(
            &mut self,
            path: &str,
            stage_flags: ash::vk::ShaderStageFlags,
        ) -> Resolvable<ShaderIx> {
            let cell = Arc::new(AtomicCell::new(None));
            let inner = cell.clone();

            let shader_path = path.to_string();

            let resolver = Box::new(
                move |_ctx: &VkContext,
                      res: &mut GpuResources,
                      _alloc: &mut Allocator| {
                    let shader = res.load_shader(&shader_path, stage_flags)?;

                    inner.store(Some(shader));
                    Ok(())
                },
            ) as ResolverFn;

            let priority = Priority::primary(ResolveOrder::Shader);

            self.resolvers.entry(priority).or_default().push(resolver);

            Resolvable {
                priority,
                value: cell,
            }
        }

        pub fn create_desc_set(
            &mut self,
            stage_flags: ash::vk::ShaderStageFlags,
            set_info: BTreeMap<u32, rspirv_reflect::DescriptorInfo>,
            inputs: Vec<rhai::Map>,
        ) -> Resolvable<DescSetIx> {
            let cell = Arc::new(AtomicCell::new(None));
            let inner = cell.clone();

            let resolver = Box::new(
                move |ctx: &VkContext,
                      res: &mut GpuResources,
                      alloc: &mut Allocator| {
                    let desc_set = res.allocate_desc_set_dyn(
                        stage_flags,
                        &set_info,
                        &inputs,
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
