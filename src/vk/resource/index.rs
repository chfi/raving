use ash::{vk, Device};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};

use thunderdome::{Arena, Index};

use zerocopy::{AsBytes, FromBytes};

use super::{BufferRes, GpuResources, ImageRes, ShaderInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderIx(pub(crate) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineIx(pub(crate) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DescSetIx(pub Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageIx(pub(crate) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageViewIx(pub(crate) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SamplerIx(pub(crate) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferIx(pub Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemaphoreIx(pub(crate) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FenceIx(pub Index);

impl std::ops::Index<ShaderIx> for GpuResources {
    type Output = ShaderInfo;

    fn index(&self, i: ShaderIx) -> &Self::Output {
        &self.shaders[i.0]
    }
}

impl std::ops::Index<PipelineIx> for GpuResources {
    type Output = (vk::Pipeline, vk::PipelineLayout);

    fn index(&self, i: PipelineIx) -> &Self::Output {
        &self.pipelines[i.0]
    }
}

impl std::ops::Index<DescSetIx> for GpuResources {
    type Output = vk::DescriptorSet;

    fn index(&self, i: DescSetIx) -> &Self::Output {
        &self.descriptor_sets[i.0]
    }
}

impl std::ops::Index<ImageIx> for GpuResources {
    type Output = ImageRes;

    fn index(&self, i: ImageIx) -> &Self::Output {
        &self.images[i.0]
    }
}
impl std::ops::IndexMut<ImageIx> for GpuResources {
    fn index_mut(&mut self, i: ImageIx) -> &mut Self::Output {
        &mut self.images[i.0]
    }
}

impl std::ops::Index<SamplerIx> for GpuResources {
    type Output = vk::Sampler;

    fn index(&self, i: SamplerIx) -> &Self::Output {
        &self.samplers[i.0]
    }
}

impl std::ops::Index<ImageViewIx> for GpuResources {
    type Output = vk::ImageView;

    fn index(&self, i: ImageViewIx) -> &Self::Output {
        &self.image_views[i.0]
    }
}

impl std::ops::Index<BufferIx> for GpuResources {
    type Output = BufferRes;

    fn index(&self, i: BufferIx) -> &Self::Output {
        &self.buffers[i.0]
    }
}

impl std::ops::IndexMut<BufferIx> for GpuResources {
    fn index_mut(&mut self, i: BufferIx) -> &mut Self::Output {
        &mut self.buffers[i.0]
    }
}

impl std::ops::Index<SemaphoreIx> for GpuResources {
    type Output = vk::Semaphore;

    fn index(&self, i: SemaphoreIx) -> &Self::Output {
        &self.semaphores[i.0]
    }
}

impl std::ops::Index<FenceIx> for GpuResources {
    type Output = vk::Fence;

    fn index(&self, i: FenceIx) -> &Self::Output {
        &self.fences[i.0]
    }
}
