use ash::{vk, Device};

use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};
use rustc_hash::FxHashMap;
use thunderdome::Arena;

use super::super::{
    context::VkContext,
    descriptor::{
        BindingDesc, BindingInput, DescriptorAllocator, DescriptorBuilder,
        DescriptorLayoutCache,
    },
};
use super::*;

/// V corresponds to the vertex index in the graph
#[derive(Default, Clone)]
pub struct GraphData<V: Copy + Eq> {
    pub images: FxHashMap<ImageIx, V>,
    pub image_views: FxHashMap<ImageViewIx, V>,

    pub semaphores: FxHashMap<SemaphoreIx, V>,

    pub desc_sets: FxHashMap<DescSetIx, V>,
}

impl<V: Copy + Eq> GraphData<V> {
    // pub fn
}
