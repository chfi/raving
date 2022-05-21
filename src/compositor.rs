use crate::vk::context::VkContext;
use crate::vk::{
    BufferIx, DescSetIx, FramebufferIx, GpuResources, PipelineIx, RenderPassIx,
    ShaderIx, VkEngine,
};
use crossbeam::atomic::AtomicCell;
use parking_lot::RwLock;

use crate::vk::resource::WindowResources;

use ash::{vk, Device};

use rhai::plugin::RhaiResult;
use rustc_hash::{FxHashMap, FxHashSet};
use winit::event::VirtualKeyCode;
use winit::window::Window;

use std::collections::{BTreeMap, HashMap};

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use zerocopy::{AsBytes, FromBytes};

use rhai::plugin::*;

pub mod label_space;

use label_space::LabelSpace;

#[derive(Clone)]
pub struct SublayerAllocMsg {
    layer_name: rhai::ImmutableString,

    sublayer_name: rhai::ImmutableString,
    sublayer_def: rhai::ImmutableString,

    sets: Vec<DescSetIx>,
}

impl SublayerAllocMsg {
    pub fn new(
        layer_name: &str,
        sublayer_name: &str,
        sublayer_def: &str,
        sets: &[DescSetIx],
    ) -> Self {
        Self {
            layer_name: layer_name.into(),
            sublayer_name: sublayer_name.into(),
            sublayer_def: sublayer_def.into(),

            sets: sets.to_vec(),
        }
    }
}

pub struct Compositor {
    window_dims: Arc<AtomicCell<[u32; 2]>>,
    pub sublayer_defs: BTreeMap<rhai::ImmutableString, SublayerDef>,

    pub clear_pass: RenderPassIx,
    pub load_pass: RenderPassIx,

    pub layers: Arc<RwLock<BTreeMap<rhai::ImmutableString, Layer>>>,

    pub sublayer_alloc_tx: crossbeam::channel::Sender<SublayerAllocMsg>,
    sublayer_alloc_rx: crossbeam::channel::Receiver<SublayerAllocMsg>,
}

impl Compositor {
    pub fn init(
        engine: &mut VkEngine,
        window_dims: &Arc<AtomicCell<[u32; 2]>>,
        init_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
        // sublayer_defs: F,
        // font_desc_set: DescSetIx,
    ) -> Result<Self> {
        let mut sublayer_defs = BTreeMap::default();

        let (clear_pass, load_pass) =
            engine.with_allocators(|ctx, res, alloc| {
                let format = vk::Format::R8G8B8A8_UNORM;
                let clear_pass = res.create_render_pass(
                    ctx,
                    format,
                    init_layout,
                    final_layout,
                    true,
                )?;

                let load_pass = res.create_render_pass(
                    ctx,
                    format,
                    init_layout,
                    final_layout,
                    false,
                )?;

                let clear_pass_ix = res.insert_render_pass(clear_pass);
                let load_pass_ix = res.insert_render_pass(load_pass);

                Ok((clear_pass_ix, load_pass_ix))
            })?;

        let layers = Arc::new(RwLock::new(BTreeMap::default()));
        // let layer_priority = BTreeMap::default();

        let (tx, rx) = crossbeam::channel::unbounded();

        Ok(Self {
            window_dims: window_dims.clone(),
            sublayer_defs,

            clear_pass,
            load_pass,

            // layer,
            layers,

            sublayer_alloc_tx: tx,
            sublayer_alloc_rx: rx,
        })
    }

    pub fn toggle_layer(&self, layer_name: &str, enabled: bool) {
        let mut layers = self.layers.write();
        if let Some(layer) = layers.get_mut(layer_name) {
            layer.enabled = enabled;
        }
    }

    pub fn window_dims(&self) -> [u32; 2] {
        self.window_dims.load()
    }

    pub fn window_dims_arc(&self) -> &Arc<AtomicCell<[u32; 2]>> {
        &self.window_dims
    }

    pub fn allocate_sublayers(&mut self, engine: &mut VkEngine) -> Result<()> {
        let mut layers = self.layers.write();

        while let Ok(msg) = self.sublayer_alloc_rx.try_recv() {
            if let Some(layer) = layers.get_mut(&msg.layer_name) {
                Self::push_sublayer(
                    &self.sublayer_defs,
                    engine,
                    layer,
                    &msg.sublayer_def,
                    &msg.sublayer_name,
                    msg.sets,
                )?;
            } else {
                log::error!(
                    "tried to allocate sublayer for nonexistent layer `{}`",
                    msg.layer_name
                );
            }
        }

        Ok(())
    }

    pub fn new_layer(&self, name: &str, depth: usize, enabled: bool) {
        let layer = Layer::new(depth, enabled);
        self.layers.write().insert(name.into(), layer);
    }

    pub fn write_layers(&self, res: &mut GpuResources) -> Result<()> {
        let mut layers = self.layers.write();

        for (name, layer) in layers.iter_mut() {
            for (sub_name, ix) in layer.sublayer_names.iter() {
                let sublayer = layer.sublayers.get_mut(*ix).ok_or(anyhow!(
                    "Error getting sublayer `{}` at index `{}`, in layer `{}`",
                    sub_name,
                    ix,
                    name,
                ))?;

                let def_name = sublayer.def_name.clone();

                sublayer.write_buffer(res).ok_or(anyhow!(
                    "Error writing sublayer buffer in layer `{}`, \
sublayer `{}`, sublayer def `{}`",
                    name,
                    sub_name,
                    def_name
                ))?;
            }
        }

        Ok(())
    }

    pub fn add_sublayer_defs(
        &mut self,
        sublayer_defs: impl IntoIterator<Item = SublayerDef>,
    ) {
        for def in sublayer_defs {
            self.sublayer_defs.insert(def.name.clone(), def);
        }
    }

    /*
    pub fn draw_into_image(
        &self,
        engine: &mut VkEngine,
        dims: [u32; 2],
    ) -> Result<Vec<u8>> {
        let [width, height] = dims;

        let px_count = (width * height) as usize;

        // 3 channels of f32 colors
        let out = Vec::with_capacity(px_count * 4 * 3);

        let mut window_resources = WindowResources::new();
        window_resources.add_image(
            "out",
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC,
            [
                (vk::ImageUsageFlags::STORAGE, vk::ImageLayout::GENERAL),
                (vk::ImageUsageFlags::SAMPLED, vk::ImageLayout::GENERAL),
            ],
            Some(self.pass),
        )?;

        let builder = window_resources.build(engine, width, height)?;

        builder.insert(
            &mut window_resources.indices,
            &engine.context,
            &mut engine.resources,
            &mut engine.allocator,
        )?;

        window_resources.destroy(
            &engine.context,
            &mut engine.resources,
            &mut engine.allocator,
        )?;

        Ok(out)
    }
    */

    pub fn draw<'a>(
        &'a self,
        clear_color: Option<[f32; 3]>,
        framebuffer: FramebufferIx,
        extent: vk::Extent2D,
    ) -> Box<dyn Fn(&Device, &GpuResources, vk::CommandBuffer) + 'a> {
        let clear_values = clear_color.map(|color| {
            [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [color[0], color[1], color[2], 1.0],
                },
            }]
        });

        let draw = move |device: &Device,
                         res: &GpuResources,
                         cmd: vk::CommandBuffer| {
            let pass_info = {
                let pass_info = vk::RenderPassBeginInfo::builder()
                    .framebuffer(res[framebuffer])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    });

                if let Some(cv) = clear_values.as_ref() {
                    pass_info
                        .render_pass(res[self.clear_pass])
                        .clear_values(cv)
                        .build()
                } else {
                    pass_info
                        .render_pass(res[self.load_pass])
                        .clear_values(&[])
                        .build()
                }
            };

            unsafe {
                device.cmd_begin_render_pass(
                    cmd,
                    &pass_info,
                    vk::SubpassContents::INLINE,
                );

                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };

                let viewports = [viewport];

                device.cmd_set_viewport(cmd, 0, &viewports);

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                };
                let scissors = [scissor];

                device.cmd_set_scissor(cmd, 0, &scissors);

                let sublayers = {
                    let layers = self.layers.read();

                    let mut layer_vec = layers
                        .iter()
                        .filter_map(|(_, layer)| {
                            if layer.enabled {
                                Some((
                                    layer.depth,
                                    layer
                                        .sublayer_order
                                        .iter()
                                        .map(|i| &layer.sublayers[*i]),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    layer_vec.sort_by_key(|(depth, _)| *depth);

                    layer_vec
                        .into_iter()
                        .flat_map(|(_, sublayer)| {
                            sublayer.map(|sublayer| {
                                let def_name = sublayer.def_name.clone();
                                let vertices = sublayer.vertex_buffer;
                                let vx_count = sublayer.vertex_count;
                                let i_count = sublayer.instance_count;
                                let sets = sublayer.sets.clone();

                                (def_name, vertices, vx_count, i_count, sets)
                            })
                        })
                        .collect::<Vec<_>>()

                    // let mut layer_vec = layers.iter().collect::<Vec<_>>();
                    // layer_vec.sort_by_key(|(_, l)| *l.depth);
                };

                for (def_name, vertices, vx_count, i_count, sets) in sublayers {
                    log::trace!("drawing sublayer {}", def_name);
                    let def = self.sublayer_defs.get(&def_name).unwrap();

                    // let vertices = sublayer.vertex_buffer;
                    // let sets = sublayer.sets.iter().copied();
                    // let vx_count = sublayer.vertex_count;
                    // let i_count = sublayer.instance_count;

                    def.draw(
                        vertices,
                        vx_count,
                        i_count,
                        sets,
                        clear_color,
                        extent,
                        device,
                        res,
                        cmd,
                    );
                }

                device.cmd_end_render_pass(cmd);
            }
        };

        Box::new(draw)
    }

    pub fn with_layer<T, F>(&self, layer_name: &str, f: F) -> Result<T>
    where
        F: FnOnce(&mut Layer) -> Result<T>,
    {
        let mut layers = self.layers.write();
        let layer = layers
            .get_mut(layer_name)
            .ok_or(anyhow!("Layer `{}` not found", layer_name))?;
        f(layer)
    }

    pub fn push_sublayer(
        defs: &BTreeMap<rhai::ImmutableString, SublayerDef>,
        engine: &mut VkEngine,
        layer: &mut Layer,
        def_name: &str,
        sublayer_name: &str,
        sets: impl IntoIterator<Item = DescSetIx>,
    ) -> Result<()> {
        let def = defs.get(def_name).ok_or(anyhow!(
            "could not find sublayer definition `{}`",
            def_name
        ))?;

        let capacity = 1024 * 1024;

        let vertex_buffer = engine.with_allocators(|ctx, res, alloc| {
            let mem_loc = gpu_allocator::MemoryLocation::CpuToGpu;
            let usage = vk::BufferUsageFlags::VERTEX_BUFFER
                // | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = res.allocate_buffer(
                ctx,
                alloc,
                mem_loc,
                def.vertex_stride,
                capacity,
                usage,
                Some(&format!(
                    "Buffer: Sublayer {}.{}",
                    def_name, sublayer_name
                )),
            )?;

            let buf_ix = res.insert_buffer(buffer);

            Ok(buf_ix)
        })?;

        let sublayer = Sublayer {
            def_name: def.name.clone(),

            instance_count: def.default_instance_count.unwrap_or_default(),
            vertex_count: def.default_vertex_count.unwrap_or_default(),
            per_instance: def.per_instance,

            vertex_stride: def.vertex_stride,
            vertex_data: Vec::new(),
            vertex_buffer,

            sets: sets.into_iter().collect(),

            need_write: false,
        };

        let i = layer.sublayers.len();

        let name = rhai::ImmutableString::from(sublayer_name);
        layer.sublayer_names.insert(name.clone(), i);
        layer.sublayer_order.push(i);
        layer.sublayers.push(sublayer);

        Ok(())
    }

    /*
    pub fn push_sublayer(
        &mut self,
        engine: &mut VkEngine,
        def_name: &str,
        sets: impl IntoIterator<Item = DescSetIx>,
    ) -> Result<()> {
        let def = self.sublayer_defs.get(def_name).ok_or(anyhow!(
            "could not find sublayer definition `{}`",
            def_name
        ))?;

        let capacity = 1024 * 1024;

        let vertex_buffer = engine.with_allocators(|ctx, res, alloc| {
            let mem_loc = gpu_allocator::MemoryLocation::CpuToGpu;
            let usage = vk::BufferUsageFlags::VERTEX_BUFFER
                // | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = res.allocate_buffer(
                ctx,
                alloc,
                mem_loc,
                def.vertex_stride,
                capacity,
                usage,
                Some(&format!("sublayer {}", def_name)),
            )?;

            let buf_ix = res.insert_buffer(buffer);

            Ok(buf_ix)
        })?;

        let sublayer = Sublayer {
            def_name: def.name.clone(),

            instance_count: def.default_instance_count.unwrap_or_default(),
            vertex_count: def.default_vertex_count.unwrap_or_default(),
            per_instance: def.per_instance,

            vertex_stride: def.vertex_stride,
            vertex_data: Vec::new(),
            vertex_buffer,

            sets: sets.into_iter().collect(),
        };

        self.layer.sublayers.push(sublayer);

        Ok(())
    }
    */
}

#[derive(Clone, Default)]
pub struct Layer {
    pub sublayers: Vec<Sublayer>,
    pub sublayer_order: Vec<usize>,
    pub sublayer_names: BTreeMap<rhai::ImmutableString, usize>,

    pub depth: usize,
    pub enabled: bool,
}

impl Layer {
    pub fn new(depth: usize, enabled: bool) -> Self {
        Layer {
            sublayers: Vec::new(),
            sublayer_order: Vec::new(),
            sublayer_names: BTreeMap::default(),

            depth,
            enabled,
        }
    }

    pub fn get_sublayer_mut(&mut self, name: &str) -> Option<&mut Sublayer> {
        let ix = *self.sublayer_names.get(name)?;
        self.sublayers.get_mut(ix)
    }
}

#[derive(Clone)]
pub struct Sublayer {
    pub def_name: rhai::ImmutableString,

    vertex_stride: usize,

    vertex_count: usize,
    instance_count: usize,
    per_instance: bool,

    vertex_data: Vec<u8>,

    vertex_buffer: BufferIx,

    sets: Vec<DescSetIx>,

    need_write: bool,
}

impl Sublayer {
    pub fn update_sets(
        &mut self,
        new_sets: impl IntoIterator<Item = DescSetIx>,
    ) {
        self.sets.clear();
        self.sets.extend(new_sets);
    }

    pub fn update_vertices_raw(
        &mut self,
        data: &[u8],
        vertex_count: usize,
        instance_count: usize,
    ) {
        // assert!(data.len() % vertex_count == 0);
        self.vertex_data.clear();
        self.vertex_data.extend_from_slice(data);
        self.vertex_count = vertex_count;
        self.instance_count = instance_count;

        self.need_write = true;
    }

    /// Overwrites the vertices in the given range with the values
    /// from the iterator.
    ///
    /// If the iterator produces fewer values than elements in
    /// `range`, the remaining vertices aren't updated; if more values
    /// are produced, they are discarded.
    pub fn update_vertices_array_range<const N: usize, I>(
        &mut self,
        range: std::ops::Range<usize>,
        new: I,
    ) -> Result<()>
    where
        I: IntoIterator<Item = [u8; N]>,
    {
        assert!(N == self.vertex_stride);

        let bytes_len = range.end * self.vertex_stride;
        self.vertex_data.resize(bytes_len, 0);
        if self.per_instance {
            self.instance_count = range.end;
        } else {
            self.vertex_count = range.end;
        }
        self.need_write = true;

        let bytes_range = (range.start * N)..(range.end * N);

        for (src, dst) in new
            .into_iter()
            .zip(self.vertex_data[bytes_range].chunks_exact_mut(N))
        {
            dst.clone_from_slice(&src);
        }

        Ok(())
    }

    /// overwrites the vertices after the provided offset with the
    /// vertices from the given iterator
    pub fn update_vertices_array_offset<const N: usize, I, J>(
        &mut self,
        offset: usize,
        new: I,
    ) -> Result<()>
    where
        I: IntoIterator<IntoIter = J>,
        J: ExactSizeIterator + Iterator<Item = [u8; N]>,
    {
        assert!(N == self.vertex_stride);

        let start_byte = offset * N;

        self.vertex_data.truncate(start_byte);
        if self.per_instance {
            self.instance_count = offset;
        } else {
            self.vertex_count = offset;
        }
        self.need_write = true;

        for slice in new.into_iter() {
            self.vertex_data.extend_from_slice(&slice);

            if self.per_instance {
                self.instance_count += 1;
            } else {
                self.vertex_count += 1;
            }
        }

        Ok(())
    }

    pub fn update_vertices_array<const N: usize, I>(
        &mut self,
        new: I,
    ) -> Result<()>
    where
        I: IntoIterator<Item = [u8; N]>,
    {
        assert!(N == self.vertex_stride);
        if self.per_instance {
            self.vertex_data.clear();
            self.instance_count = 0;
        } else {
            self.vertex_data.clear();
            self.vertex_count = 0;
        }
        self.need_write = true;

        for slice in new.into_iter() {
            self.vertex_data.extend_from_slice(&slice);

            if self.per_instance {
                self.instance_count += 1;
            } else {
                self.vertex_count += 1;
            }
        }

        Ok(())
    }

    pub fn update_vertices<'a, I>(&mut self, new: I) -> Result<()>
    where
        I: IntoIterator<Item = &'a [u8]> + 'a,
    {
        if self.per_instance {
            self.vertex_data.clear();
            self.instance_count = 0;
        } else {
            self.vertex_data.clear();
            self.vertex_count = 0;
        }
        self.need_write = true;

        for slice in new.into_iter() {
            if slice.len() != self.vertex_stride {
                anyhow::bail!(
                    "slice length {} doesn't match vertex stride {}",
                    slice.len(),
                    self.vertex_stride
                );
            }

            self.vertex_data.extend_from_slice(slice);

            if self.per_instance {
                self.instance_count += 1;
            } else {
                self.vertex_count += 1;
            }
        }

        Ok(())
    }

    pub fn write_buffer(&mut self, res: &mut GpuResources) -> Option<()> {
        if !self.need_write {
            return Some(());
        }
        assert!(self.vertex_data.len() % self.vertex_stride == 0);

        let buf = &mut res[self.vertex_buffer];
        let slice = buf.mapped_slice_mut()?;
        let len = self.vertex_data.len();
        slice[0..len].clone_from_slice(&self.vertex_data);
        self.need_write = false;
        Some(())
    }
}

pub struct SublayerDef {
    pub name: rhai::ImmutableString,

    pub(super) clear_pipeline: PipelineIx,
    pub(super) load_pipeline: PipelineIx,

    pub(super) sets: Vec<DescSetIx>,
    pub(super) vertex_stride: usize,

    per_instance: bool,

    vertex_offset: usize,
    default_vertex_count: Option<usize>,
    default_instance_count: Option<usize>,

    elem_type: std::any::TypeId,
}

impl SublayerDef {
    pub fn draw(
        &self,
        vertices: BufferIx,
        vertex_count: usize,
        instance_count: usize,
        sets: impl IntoIterator<Item = DescSetIx>,
        clear_color: Option<[f32; 3]>,
        extent: vk::Extent2D,
        device: &Device,
        res: &GpuResources,
        cmd: vk::CommandBuffer,
    ) {
        let (pipeline, layout) = if clear_color.is_some() {
            res[self.clear_pipeline]
        } else {
            res[self.load_pipeline]
        };

        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            let vx_buf_ix = vertices;
            let vx_buf = res[vx_buf_ix].buffer;
            let vxs = [vx_buf];
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                &vxs,
                &[self.vertex_offset as u64],
            );

            let dims = [extent.width as f32, extent.height as f32];
            let constants = bytemuck::cast_slice(&dims);

            let stages =
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            device.cmd_push_constants(cmd, layout, stages, 0, constants);

            let descriptor_sets = self
                .sets
                .iter()
                .copied()
                .chain(sets.into_iter())
                .map(|s| res[s])
                .collect::<Vec<_>>();

            if !descriptor_sets.is_empty() {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &descriptor_sets,
                    &[],
                );
            }

            device.cmd_draw(
                cmd,
                vertex_count as u32,
                instance_count as u32,
                0,
                0,
            );
        }
    }

    pub fn new<'a, T, S>(
        ctx: &VkContext,
        res: &mut GpuResources,
        name: &str,
        vert: ShaderIx,
        frag: ShaderIx,
        clear_pass: vk::RenderPass,
        load_pass: vk::RenderPass,
        vertex_offset: usize,
        vertex_stride: usize,
        per_instance: bool,
        default_vertex_count: Option<usize>,
        default_instance_count: Option<usize>,
        vert_input_info: vk::PipelineVertexInputStateCreateInfoBuilder<'a>,
        sets: S,
    ) -> Result<Self>
    where
        // T: std::any::Any + Copy + AsBytes,
        S: IntoIterator<Item = DescSetIx>,
        T: std::any::Any + Copy,
    {
        let clear_pipeline = res.create_graphics_pipeline(
            ctx,
            vert,
            frag,
            clear_pass,
            &vert_input_info,
        )?;

        let load_pipeline = res.create_graphics_pipeline(
            ctx,
            vert,
            frag,
            load_pass,
            &vert_input_info,
        )?;

        {
            let (pipeline, pipeline_layout) = res[clear_pipeline];

            VkEngine::set_debug_object_name(
                ctx,
                pipeline,
                &format!("Sublayer CLEAR Pipeline: `{}`", name),
            )?;
            VkEngine::set_debug_object_name(
                ctx,
                pipeline_layout,
                &format!("Sublayer CLEAR Pipeline Layout: `{}`", name),
            )?;

            let (pipeline, pipeline_layout) = res[load_pipeline];

            VkEngine::set_debug_object_name(
                ctx,
                pipeline,
                &format!("Sublayer LOAD Pipeline: `{}`", name),
            )?;
            VkEngine::set_debug_object_name(
                ctx,
                pipeline_layout,
                &format!("Sublayer LOAD Pipeline Layout: `{}`", name),
            )?;
        }

        Ok(Self {
            name: name.into(),

            clear_pipeline,
            load_pipeline,

            sets: sets.into_iter().collect(),
            vertex_stride,
            vertex_offset,
            per_instance,

            default_vertex_count,
            default_instance_count,

            elem_type: std::any::TypeId::of::<T>(),
        })
    }
}

pub fn create_rhai_module(
    compositor: &Compositor,
    // buffers: &BufferStorage,
) -> rhai::Module {
    let mut module: rhai::Module = rhai::exported_module!(rhai_module);

    let layers = compositor.layers.clone();

    module.set_native_fn(
        "init_layer",
        move |name: &str, depth: i64, enabled: bool| {
            let mut layers = layers.write();
            if layers.contains_key(name) {
                return Ok(rhai::Dynamic::FALSE);
            }

            layers.insert(name.into(), Layer::new(depth as usize, enabled));

            Ok(rhai::Dynamic::TRUE)
        },
    );

    let alloc_tx = compositor.sublayer_alloc_tx.clone();

    module.set_native_fn(
        "allocate_sublayer",
        move |layer_name: &str, sublayer_def: &str, sublayer_name: &str| {
            let msg = SublayerAllocMsg::new(
                layer_name,
                sublayer_name,
                sublayer_def,
                &[],
            );

            if let Err(e) = alloc_tx.send(msg) {
                Err(format!("sublayer allocation message error: {:?}", e)
                    .into())
            } else {
                Ok(())
            }
        },
    );

    let layers = compositor.layers.clone();
    module.set_native_fn(
        "toggle_layer",
        move |layer_name: &str, enabled: bool| {
            let mut layers = layers.write();
            if let Some(layer) = layers.get_mut(layer_name) {
                layer.enabled = enabled;
                return Ok(layer.enabled);
            }

            Ok(false)
        },
    );

    module
}

#[export_module]
pub mod rhai_module {

    pub type Layer = super::Layer;
    pub type Sublayer = super::Sublayer;
}
