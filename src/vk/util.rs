use crate::vk::descriptor::{BindingDesc, BindingInput};
use crate::vk::{BatchInput, GpuResources, VkEngine};

use super::resource::index::*;
use ash::{vk, Device};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};

#[derive(Clone, Copy)]
pub struct LineRenderer {
    pub text_buffer: BufferIx,
    text_len: usize,

    pub line_buffer: BufferIx,
    line_count: usize,
}

impl LineRenderer {
    pub fn update_lines<'a>(
        &mut self,
        res: &mut GpuResources,
        lines: impl IntoIterator<Item = &'a str>,
    ) -> Result<()> {
        let mut line_intervals: Vec<(usize, usize)> = Vec::new();

        let mut text_offset = 0;

        {
            let text_buffer = &mut res[self.text_buffer];

            let mapped = text_buffer
                .alloc
                .mapped_slice_mut()
                .ok_or(anyhow!("couldn't map text buffer memory!"))?;

            let mut ix = 4;
            let mut write_u32 = |u: u32| {
                for (i, b) in u.to_le_bytes().into_iter().enumerate() {
                    mapped[ix + i] = b;
                }
                ix += 4;
            };

            for text in lines.into_iter() {
                let start = text_offset;
                let end = start + text.len();
                text_offset = end;

                for &b in text.as_bytes() {
                    write_u32(b as u32);
                }

                line_intervals.push((start, end));
            }

            for (i, b) in
                (text_offset as u32).to_le_bytes().into_iter().enumerate()
            {
                mapped[i] = b;
            }

            self.text_len = text_offset;
            self.line_count = line_intervals.len();
        }

        {
            let line_buffer = &mut res[self.line_buffer];

            let mapped = line_buffer
                .alloc
                .mapped_slice_mut()
                .ok_or(anyhow!("couldn't map line buffer memory!"))?;

            let mut ix = 0;
            let mut write_u32 = |u: u32| {
                for (i, b) in u.to_le_bytes().into_iter().enumerate() {
                    mapped[ix + i] = b;
                }
                ix += 4;
            };

            for (start, end) in line_intervals {
                write_u32(start as u32);
                write_u32(end as u32);
            }
        }

        Ok(())
    }

    pub fn new(engine: &mut VkEngine) -> Result<Self> {
        let result = engine.with_allocators(|ctx, res, alloc| {
            let usage = vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER;

            let text_buffer = res.allocate_buffer(
                ctx,
                alloc,
                gpu_allocator::MemoryLocation::CpuToGpu,
                1,
                256 * 256,
                usage,
                Some("lines:text_data_buffer"),
            )?;

            let line_buffer = res.allocate_buffer(
                ctx,
                alloc,
                gpu_allocator::MemoryLocation::CpuToGpu,
                1,
                256 * 4 * 2, // 256 * 2 u32s
                usage,
                Some("lines:line_data_buffer"),
            )?;

            Ok(Self {
                text_buffer,
                text_len: 0,

                line_buffer,
                line_count: 0,
            })
        })?;

        {
            let res = &engine.resources;
            engine.set_debug_object_name(
                res[result.text_buffer].buffer,
                "lines:text_data_buffer",
            )?;
        }

        Ok(result)
    }
}

#[derive(Clone, Copy)]
pub struct ExampleState {
    pub fill_image: ImageIx,
    pub fill_view: ImageViewIx,
}

pub fn copy_batch(
    src: ImageIx,
    dst_img: vk::Image,
    device: &Device,
    resources: &GpuResources,
    cmd: vk::CommandBuffer,
) {
    let img = &resources[src];

    VkEngine::transition_image(
        cmd,
        &device,
        dst_img,
        vk::AccessFlags::NONE_KHR,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::AccessFlags::NONE_KHR,
        vk::PipelineStageFlags::TRANSFER,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    VkEngine::copy_image(
        &device,
        cmd,
        img.image,
        dst_img,
        img.extent,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    VkEngine::transition_image(
        cmd,
        &device,
        dst_img,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::MEMORY_READ,
        vk::PipelineStageFlags::HOST,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
    );
}
