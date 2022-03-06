use ash::vk;

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};

use crate::vk::{VkContext, VkEngine};

#[allow(dead_code)]
pub struct BufferRes {
    name: Option<String>,
    pub buffer: vk::Buffer,

    pub elem_size: usize,
    pub len: usize,

    pub location: gpu_allocator::MemoryLocation,

    pub alloc: Allocation,
}

impl BufferRes {
    pub fn gpu_readable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => true,
            CpuToGpu => true,
            GpuToCpu => false,
        }
    }
    pub fn gpu_writable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => true,
            CpuToGpu => false,
            GpuToCpu => true,
        }
    }

    pub fn host_readable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => false,
            CpuToGpu => true,
            GpuToCpu => true,
        }
    }

    pub fn host_writable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => false,
            CpuToGpu => true,
            GpuToCpu => false,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.len * self.elem_size
    }

    /// Returns `None` if the buffer memory can't be mapped (e.g. it's
    /// on device memory)
    pub fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        self.alloc.mapped_slice_mut()
    }

    // TODO it should probably panic if some sizes don't match
    /// Returns `None` if the buffer memory can't be mapped (e.g. it's
    /// on device memory), or if `N` does not match the element size.
    ///
    /// `unsafe` since the alignment etc. isn't fully validated when
    /// allocating a `BufferRes`, yet, but it should be safe
    pub unsafe fn mapped_windows_mut<const N: usize>(
        &mut self,
    ) -> Option<&mut [[u8; N]]> {
        if self.elem_size != N {
            return None;
        }

        let size_bytes = self.size_bytes();

        let slice = self.alloc.mapped_slice_mut()?;

        if size_bytes != slice.len() {
            return None;
        }

        let ptr = slice as *mut [u8];
        let ptr: *mut [u8; N] = ptr.cast();

        let array_slice = std::slice::from_raw_parts_mut(ptr, self.len);
        Some(array_slice)
    }

    pub fn upload_to_self_bytes(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        src: &[u8],
        cmd: vk::CommandBuffer,
    ) -> Result<Self> {
        // assert!(!self.host_writable());
        let device = ctx.device();

        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let location = MemoryLocation::CpuToGpu;

        let len = src.len().min(self.size_bytes());

        let mut staging = Self::allocate_for_type::<u8>(
            ctx,
            allocator,
            location,
            staging_usage,
            len,
            Some("tmp staging buffer"),
        )?;

        if let Some(stg) = staging.alloc.mapped_slice_mut() {
            stg[..src.len()].clone_from_slice(src);
        } else {
            bail!("couldn't map staging buffer memory");
        }

        VkEngine::copy_buffer(
            device,
            cmd,
            staging.buffer,
            self.buffer,
            len,
            None,
            None,
        );

        Ok(staging)
    }

    pub fn cleanup(
        self,
        ctx: &VkContext,
        allocator: &mut Allocator,
    ) -> Result<()> {
        unsafe {
            ctx.device().destroy_buffer(self.buffer, None);
        }

        allocator.free(self.alloc)?;
        Ok(())
    }

    pub fn allocate(
        ctx: &VkContext,
        allocator: &mut Allocator,
        location: gpu_allocator::MemoryLocation,
        usage: vk::BufferUsageFlags,
        elem_size: usize,
        len: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        let size = elem_size * len;

        let buf_info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(usage)
            .size(size as _)
            .build();

        let device = ctx.device();

        let (buffer, reqs) = unsafe {
            let buffer = device.create_buffer(&buf_info, None)?;
            let reqs = device.get_buffer_memory_requirements(buffer);
            (buffer, reqs)
        };

        let alloc_desc = AllocationCreateDesc {
            name: name.unwrap_or("tmp"),
            requirements: reqs,
            location,
            linear: true,
        };

        let alloc = allocator.allocate(&alloc_desc)?;

        unsafe {
            device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
        }?;

        Ok(Self {
            name: name.map(|n| n.to_string()),
            buffer,

            elem_size,
            len,

            location,

            alloc,
        })
    }

    pub fn allocate_for_type<T: Copy>(
        ctx: &VkContext,
        allocator: &mut Allocator,
        location: gpu_allocator::MemoryLocation,
        usage: vk::BufferUsageFlags,
        len: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        let elem_size = std::mem::size_of::<T>();
        Self::allocate(ctx, allocator, location, usage, elem_size, len, name)
    }
}
