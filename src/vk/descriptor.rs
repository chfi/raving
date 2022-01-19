use std::collections::{HashMap, VecDeque};

use ash::{
    extensions::khr::{Surface, Swapchain},
    // version::DeviceV1_0,
    vk::{self},
    Device,
    Entry,
};

use gpu_allocator::vulkan::Allocator;
use rustc_hash::FxHashMap;
use winit::window::Window;

use anyhow::{anyhow, bail, Result};

use thunderdome::{Arena, Index};

use super::context::VkContext;

pub struct DescriptorAllocator {
    device: Device,

    current_pool: Option<vk::DescriptorPool>,

    free_pools: VecDeque<vk::DescriptorPool>,
    used_pools: VecDeque<vk::DescriptorPool>,
}

pub struct DescriptorLayoutCache {
    device: Device,

    layout_cache: HashMap<DescriptorLayoutInfo, vk::DescriptorSetLayout>,
}

#[derive(Default)]
pub struct DescriptorLayoutInfo {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl PartialEq for DescriptorLayoutInfo {
    fn eq(&self, other: &Self) -> bool {
        if self.bindings.len() != other.bindings.len() {
            return false;
        }

        self.bindings
            .iter()
            .zip(other.bindings.iter())
            .all(|(&a, &b)| {
                a.binding == b.binding
                    && a.descriptor_count == b.descriptor_count
                    && a.descriptor_type == b.descriptor_type
                    && a.stage_flags == b.stage_flags
            })
    }
}

impl Eq for DescriptorLayoutInfo {}

impl std::hash::Hash for DescriptorLayoutInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for layout_binding in &self.bindings {
            layout_binding.binding.hash(state);
            layout_binding.descriptor_count.hash(state);
            layout_binding.descriptor_type.hash(state);
            layout_binding.stage_flags.hash(state);
        }
    }
}

pub struct DescriptorBuilder<'a> {
    layout_cache: &'a mut DescriptorLayoutCache,
    allocator: &'a mut DescriptorAllocator,

    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    writes: Vec<vk::WriteDescriptorSet>,
}

impl DescriptorAllocator {
    const POOL_SIZES: [(vk::DescriptorType, f32); 11] = {
        use vk::DescriptorType as Ty;
        [
            (Ty::SAMPLER, 0.5),
            (Ty::COMBINED_IMAGE_SAMPLER, 4.0),
            (Ty::SAMPLED_IMAGE, 4.0),
            (Ty::STORAGE_IMAGE, 1.0),
            (Ty::UNIFORM_TEXEL_BUFFER, 1.0),
            (Ty::STORAGE_TEXEL_BUFFER, 1.0),
            (Ty::UNIFORM_BUFFER, 2.0),
            (Ty::STORAGE_BUFFER, 2.0),
            (Ty::UNIFORM_BUFFER_DYNAMIC, 1.0),
            (Ty::STORAGE_BUFFER_DYNAMIC, 1.0),
            (Ty::INPUT_ATTACHMENT, 0.5),
        ]
    };

    pub(super) fn new(ctx: &VkContext) -> Result<Self> {
        let free_pools = Default::default();
        let used_pools = Default::default();

        Ok(Self {
            device: ctx.device().to_owned(),

            current_pool: None,

            free_pools,
            used_pools,
        })
    }

    fn create_pool(
        device: &Device,
        count: u32,
        flags: vk::DescriptorPoolCreateFlags,
    ) -> Result<vk::DescriptorPool> {
        // let fcount = count as f32;

        let sizes = Self::POOL_SIZES
            .into_iter()
            .map(|(ty, size)| {
                let desc_count = (count as f32 * size as f32) as u32;
                vk::DescriptorPoolSize::builder()
                    .descriptor_count(desc_count)
                    .ty(ty)
                    .build()
            })
            .collect::<Vec<_>>();

        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(count)
            .flags(flags)
            .pool_sizes(sizes.as_slice())
            .build();

        let pool = unsafe { device.create_descriptor_pool(&create_info, None)? };

        Ok(pool)
    }

    pub(super) fn reset_pools(&mut self) -> Result<()> {
        for &pool in &self.used_pools {
            unsafe {
                self.device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())?
            };
        }

        // TODO i think this is correct?
        self.free_pools.clone_from(&mut self.used_pools);
        self.used_pools.clear();

        self.current_pool = None;

        Ok(())
    }

    pub(super) fn allocate(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let current_pool = if let Some(pool) = self.current_pool {
            pool
        } else {
            let new_pool = self.grab_descriptor_pool()?;
            self.current_pool = Some(new_pool);
            self.used_pools.push_back(new_pool);
            new_pool
        };

        let layouts = [layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .set_layouts(&layouts)
            .descriptor_pool(current_pool)
            .build();

        let alloc_result = unsafe { self.device.allocate_descriptor_sets(&alloc_info) };

        let desc_sets = match alloc_result {
            Ok(sets) => Some(sets),
            Err(vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY) => None,
            Err(_) => bail!("Unrecoverable error when attempting to allocate descriptor set"),
        };

        if let Some(sets) = desc_sets {
            return Ok(sets[0]);
        }

        // need reallocation

        let new_pool = self.grab_descriptor_pool()?;
        self.current_pool = Some(new_pool);
        self.used_pools.push_back(new_pool);

        let alloc_result = unsafe { self.device.allocate_descriptor_sets(&alloc_info) };

        // if reallocation doesn't work, we're screwed
        match alloc_result {
            Ok(sets) => Ok(sets[0]),
            Err(_) => bail!("Unrecoverable error when attempting to allocate descriptor set"),
        }
    }

    pub(super) fn cleanup(&mut self) -> Result<()> {
        for &pool in &self.free_pools {
            unsafe { self.device.destroy_descriptor_pool(pool, None) }
        }
        for &pool in &self.used_pools {
            unsafe { self.device.destroy_descriptor_pool(pool, None) }
        }

        Ok(())
    }

    pub(super) fn grab_descriptor_pool(&mut self) -> Result<vk::DescriptorPool> {
        if let Some(pool) = self.free_pools.pop_back() {
            Ok(pool)
        } else {
            Self::create_pool(&self.device, 1000, vk::DescriptorPoolCreateFlags::empty())
        }
    }
}

impl DescriptorLayoutCache {
    pub(super) fn new(device: Device) -> Self {
        Self {
            device,
            layout_cache: Default::default(),
        }
    }

    pub(super) fn get_descriptor_layout(
        &mut self,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> Result<vk::DescriptorSetLayout> {
        let mut layout_info = DescriptorLayoutInfo::default();

        let count = info.binding_count as usize;
        layout_info.bindings.reserve(count);

        // let mut is_sorted = true;
        // let mut last_binding: Option<usize> = None;
        let mut is_sorted = true;
        let mut prev = None;

        let bindings = unsafe { std::slice::from_raw_parts::<_>(info.p_bindings, count) };

        for &b in bindings {
            if let Some(p) = prev {
                is_sorted = p < b.binding;
            }
            layout_info.bindings.push(b);
            prev = Some(b.binding);
        }

        if !is_sorted {
            layout_info.bindings.sort_by_key(|l| l.binding);
        }

        if let Some(v) = self.layout_cache.get(&layout_info) {
            return Ok(*v);
        }

        let layout = unsafe { self.device.create_descriptor_set_layout(info, None)? };

        self.layout_cache.insert(layout_info, layout);

        Ok(layout)
    }

    pub(super) fn cleanup(&mut self) -> Result<()> {
        for layout in self.layout_cache.values() {
            unsafe {
                self.device.destroy_descriptor_set_layout(*layout, None);
            }
        }

        self.layout_cache.clear();

        Ok(())
    }
}

impl<'a> DescriptorBuilder<'a> {
    pub(super) fn begin(
        layout_cache: &'a mut DescriptorLayoutCache,
        allocator: &'a mut DescriptorAllocator,
    ) -> Self {
        Self {
            layout_cache,
            allocator,

            bindings: Vec::new(),
            writes: Vec::new(),
        }
    }

    pub(super) fn bind_buffer(
        &mut self,
        binding: u32,
        buffer_info: &[vk::DescriptorBufferInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> &mut Self {
        let layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .descriptor_count(1)
            .descriptor_type(ty)
            .binding(binding)
            .stage_flags(stage_flags)
            .build();

        self.bindings.push(layout_binding);

        let write = vk::WriteDescriptorSet::builder()
            .descriptor_type(ty)
            .buffer_info(buffer_info)
            .dst_binding(binding)
            .build();

        self.writes.push(write);

        self
    }

    pub(super) fn bind_image(
        &mut self,
        binding: u32,
        image_info: &[vk::DescriptorImageInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> &mut Self {
        let layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .descriptor_count(1)
            .descriptor_type(ty)
            .binding(binding)
            .stage_flags(stage_flags)
            .build();

        self.bindings.push(layout_binding);

        let mut write = vk::WriteDescriptorSet::builder()
            .descriptor_type(ty)
            .image_info(image_info)
            .dst_binding(binding);

        log::warn!("write desc count: {}", write.descriptor_count);

        let write = write.build();

        self.writes.push(write);

        log::debug!("bindings.len() {}", self.bindings.len());
        log::debug!("writes.len() {}", self.writes.len());

        self
    }

    pub(super) fn build(mut self) -> Result<vk::DescriptorSet> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(self.bindings.as_slice())
            .build();

        let layout = self.layout_cache.get_descriptor_layout(&create_info)?;

        let set = self.allocator.allocate(layout)?;

        for write in self.writes.iter_mut() {
            write.dst_set = set;
            // write.descriptor_count = 1;
        }

        // log::warn!("

        unsafe {
            self.allocator
                .device
                .update_descriptor_sets(self.writes.as_slice(), &[]);
        }

        Ok(set)
    }
}
