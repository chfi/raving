use crate::script::console::frame::FrameBuilder;
use crate::script::console::BatchBuilder;
use crate::script::EvalResult;
use crate::vk::descriptor::DescriptorLayoutInfo;
use crate::vk::{
    BatchInput, BufferIx, DescSetIx, FrameResources, FramebufferIx,
    GpuResources, PipelineIx, RenderPassIx, VkEngine,
};
use bstr::ByteSlice;
use crossbeam::atomic::AtomicCell;
use parking_lot::RwLock;
use rspirv_reflect::DescriptorInfo;

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

#[derive(Clone)]
pub struct LabelSpace {
    name: rhai::ImmutableString,

    offsets: BTreeMap<rhai::ImmutableString, (usize, usize)>,

    text: Vec<u8>,

    capacity: usize,
    used_bytes: usize,

    pub text_buffer: BufferIx,
    pub text_set: DescSetIx,
}

impl LabelSpace {
    pub fn new(
        engine: &mut VkEngine,
        name: &str,
        capacity: usize,
    ) -> Result<Self> {
        let name = format!("label-space:{}", name);

        let (text_buffer, text_set) =
            engine.with_allocators(|ctx, res, alloc| {
                let mem_loc = gpu_allocator::MemoryLocation::CpuToGpu;
                let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST;

                let buffer = res.allocate_buffer(
                    ctx,
                    alloc,
                    mem_loc,
                    4,
                    capacity / 4,
                    usage,
                    Some(&name),
                )?;

                let buf_ix = res.insert_buffer(buffer);

                let desc_set = allocate_buffer_desc_set(buf_ix, res)?;

                let set_ix = res.insert_desc_set(desc_set);

                Ok((buf_ix, set_ix))
            })?;

        Ok(Self {
            name: name.into(),

            offsets: BTreeMap::default(),
            text: Vec::new(),

            capacity,
            used_bytes: 0,

            text_buffer,
            text_set,
        })
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.used_bytes = 0;
    }

    pub fn write_buffer(&self, res: &mut GpuResources) -> Option<()> {
        if self.used_bytes == 0 {
            return Some(());
        }
        let buf = &mut res[self.text_buffer];
        let slice = buf.mapped_slice_mut()?;
        slice[0..self.used_bytes].clone_from_slice(&self.text);
        Some(())
    }

    pub fn insert(&mut self, text: &str) -> Result<()> {
        self.bounds_for_insert(text)?;
        Ok(())
    }

    pub fn bounds_for(&self, text: &str) -> Option<(usize, usize)> {
        self.offsets.get(text).copied()
    }

    pub fn bounds_for_insert(&mut self, text: &str) -> Result<(usize, usize)> {
        if let Some(bounds) = self.offsets.get(text) {
            return Ok(*bounds);
        }

        let offset = self.used_bytes;
        let len = text.as_bytes().len();

        if self.used_bytes + len > self.capacity {
            anyhow::bail!("Label space out of memory");
        }

        let bounds = (offset, len);

        self.text.extend(text.as_bytes());
        self.offsets.insert(text.into(), bounds);

        self.used_bytes += len;

        Ok(bounds)
    }
}

pub fn label_rects(
    label_space: &mut Arc<RwLock<LabelSpace>>,
    labels: rhai::Array,
) -> EvalResult<Vec<[u8; 4 * 8]>> {
    let mut space = label_space.write();

    let mut result = Vec::with_capacity(labels.len());

    let get_f32 = |map: &rhai::Map, k: &str| -> EvalResult<f32> {
        map.get(k)
            .and_then(|v| v.as_float().ok())
            .ok_or_else(|| format!("map key `{}` must be a float", k).into())
    };

    for label in labels {
        let mut map = label
            .try_cast::<rhai::Map>()
            .ok_or("array elements must be maps")?;

        let x = get_f32(&map, "x")?;
        let y = get_f32(&map, "y")?;

        let color = [
            get_f32(&map, "r")?,
            get_f32(&map, "g")?,
            get_f32(&map, "b")?,
            get_f32(&map, "a")?,
        ];

        let text = map
            .remove("contents")
            .and_then(|v| v.into_string().ok())
            .ok_or("`contents` key must be a string")?;

        let (s, l) = space.bounds_for_insert(&text).unwrap();

        let mut vertex = [0u8; 4 * 8];
        vertex[0..8].clone_from_slice([x, y].as_bytes());
        vertex[8..16].clone_from_slice([s as u32, l as u32].as_bytes());
        vertex[16..32].clone_from_slice(color.as_bytes());
        result.push(vertex);
    }

    Ok(result)
}

fn allocate_buffer_desc_set(
    buffer: BufferIx,
    res: &mut GpuResources,
) -> Result<vk::DescriptorSet> {
    // TODO also do uniforms if/when i add them, or keep them in a
    // separate set
    let layout_info = {
        let mut info = DescriptorLayoutInfo::default();

        let binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(
                vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT,
            ) // TODO should also be graphics, probably
            .build();

        info.bindings.push(binding);
        info
    };

    let set_info = {
        let info = DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_BUFFER,
            binding_count: rspirv_reflect::BindingCount::One,
            name: "samples".to_string(),
        };

        Some((0u32, info)).into_iter().collect::<BTreeMap<_, _>>()
    };

    res.allocate_desc_set_raw(&layout_info, &set_info, |res, builder| {
        let buffer = &res[buffer];
        let info = ash::vk::DescriptorBufferInfo::builder()
            .buffer(buffer.buffer)
            .offset(0)
            .range(ash::vk::WHOLE_SIZE)
            .build();
        let buffer_info = [info];
        builder.bind_buffer(0, &buffer_info);
        Ok(())
    })
}
