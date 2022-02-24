use ash::vk;
use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;

use rhai::NativeCallContext;
use rspirv_reflect::Reflection;

use std::{any::TypeId, sync::Arc};

use crate::vk::{
    context::VkContext, resource::index::*, BufferRes, GpuResources, VkEngine,
};

pub mod frame;

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

    staging_buffers: Arc<Mutex<Vec<BufferRes>>>,

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

        for buffer in buffers.drain(..) {
            res.free_buffer(ctx, alloc, buffer)?;
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

                staging_bufs.lock().push(staging);

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

                staging_bufs.lock().push(staging);

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
    engine.register_type_with_name::<ShaderIx>("ShaderIx");
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

    engine.register_result_fn(
        "dispatch_compute",
        |builder: &mut BatchBuilder,
         pipeline: PipelineIx,
         desc_set: DescSetIx,
         push_constants: Vec<u8>,
         groups: rhai::Map| {
             let get = |name: &str| {
                 let field = groups.get(name).ok_or(
"`groups` map must have integer fields `x_groups`, `y_groups`, `z_groups`".into())?;
                 field.as_int()
             };

             builder.dispatch_compute(
                 pipeline,
                 desc_set,
                 push_constants,
                 get("x_groups")?,
                 get("y_groups")?,
                 get("z_groups")?
             );
             Ok(())
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
