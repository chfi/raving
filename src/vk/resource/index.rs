use ash::{vk, Device};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};

use thunderdome::{Arena, Index};

use super::{GpuResources, ImageRes};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineIx(pub(super) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DescSetIx(pub(super) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageIx(pub(super) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageViewIx(pub(super) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferIx(pub(super) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemaphoreIx(pub(super) Index);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FenceIx(pub(super) Index);

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

impl std::ops::Index<ImageViewIx> for GpuResources {
    type Output = (vk::ImageView, ImageIx);

    fn index(&self, i: ImageViewIx) -> &Self::Output {
        &self.image_views[i.0]
    }
}

/* buffers aren't in yet
impl std::ops::Index<BufferIx> for GpuResources {
    type Output = BufferRes;

    fn index(&self, i: BufferIx) -> &Self::Output {
        &self.buffers[i.0]
    }
}
*/

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
