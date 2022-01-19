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

    free_pools: VecDeque<vk::DescriptorPool>,
    used_pools: VecDeque<vk::DescriptorPool>,
}

pub struct DescriptorLayoutCache {
    device: Device,

    layout_cache: HashMap<DescriptorLayoutInfo, vk::DescriptorSetLayout>,
}

pub struct DescriptorLayoutInfo {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl PartialEq for DescriptorLayoutInfo {
    fn eq(&self, other: &Self) -> bool {
        unimplemented!();
    }
}

impl Eq for DescriptorLayoutInfo {}

impl std::hash::Hash for DescriptorLayoutInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        todo!()
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

    pub(super) fn init(ctx: &VkContext) -> Result<Self> {
        let free_pools = Default::default();
        let used_pools = Default::default();

        Ok(Self {
            device: ctx.device().to_owned(),

            free_pools,
            used_pools,
        })
    }

    pub(super) fn reset_pools(&mut self) -> Result<()> {
        unimplemented!();
    }

    pub(super) fn allocate(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        unimplemented!();
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

    pub(super) fn grab_descriptor_pool(&mut self) -> Option<vk::DescriptorPool> {
        /*
        if let Some(pool) = self.free_pools.pop() {
            return Some(pool);
        } else {
            // Self::create_pool(self.device, self.descriptor_sizes, 1000, 0)
            todo!();
        }
        */

        unimplemented!();
    }
}

impl DescriptorLayoutCache {
    pub(super) fn init(device: Device) -> Self {
        Self {
            device,
            layout_cache: Default::default(),
        }
    }

    pub(super) fn create_descriptor_layout(
        info: vk::DescriptorSetLayoutCreateInfo,
    ) -> Result<vk::DescriptorSetLayout> {
        unimplemented!()
    }

    pub(super) fn cleanup(&mut self) -> Result<()> {
        unimplemented!();
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
        mut self,
        binding: u32,
        buffer_info: &[vk::DescriptorBufferInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(ty)
            .binding(binding)
            .stage_flags(stage_flags)
            .build();

        self.bindings.push(layout_binding);

        // TODO this is just to make sure the info lifetimes don't get messed up
        let infos = Vec::from(buffer_info);

        let write = vk::WriteDescriptorSet::builder()
            .descriptor_type(ty)
            .buffer_info(infos.as_slice()) // see above
            .dst_binding(binding)
            .build();

        self.writes.push(write);

        self
    }

    pub(super) fn bind_image(
        mut self,
        binding: u32,
        image_info: &[vk::DescriptorImageInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(ty)
            .binding(binding)
            .stage_flags(stage_flags)
            .build();

        self.bindings.push(layout_binding);

        // TODO this is just to make sure the info lifetimes don't get messed up
        let infos = Vec::from(image_info);

        let write = vk::WriteDescriptorSet::builder()
            .descriptor_type(ty)
            .image_info(infos.as_slice()) // see above
            .dst_binding(binding)
            .build();

        self.writes.push(write);

        self
    }

    pub(super) fn build(self) -> Result<()> {
        unimplemented!();
    }
}
