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

pub mod index;

pub use index::*;

#[derive(Debug, Clone, Copy)]
struct RenderPassInfo {
    clear_pass: RenderPassIx,
    load_pass: RenderPassIx,
    depth_format: Option<vk::Format>,
}

/*
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceType {
    TriMesh {
    },
    SDF,
}
*/

pub struct SceneObj {
    material: MaterialId,

    vertices: BufferIx,
}

pub struct MaterialDef {
    pub id: MaterialId,
    pub name: rhai::ImmutableString,

    // TODO add some sort of attachment and desc set data
    per_instance: bool,

    vertex_offset: usize,
    vertex_stride: usize,

    default_vertex_count: Option<usize>,
    default_instance_count: Option<usize>,
}

pub struct Scene {
    // cameras:
}

pub struct SceneCompositor {
    render_passes: HashMap<rhai::ImmutableString, RenderPassInfo>,
}

impl SceneCompositor {
    pub fn init(
        engine: &mut VkEngine,
        init_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Result<Self> {
        let mut scomp = Self {
            render_passes: HashMap::default(),
        };

        let color_format = vk::Format::R8G8B8A8_UNORM;

        scomp.register_render_pass(
            "scene-compositor",
            engine,
            init_layout,
            final_layout,
            color_format,
            None,
        )?;

        let depth_format = GpuResources::find_supported_format(
            engine.ctx(),
            [
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;

        scomp.register_render_pass(
            "scene-compositor-depth",
            engine,
            init_layout,
            final_layout,
            color_format,
            Some(depth_format),
        )?;

        Ok(scomp)
    }

    pub fn register_render_pass(
        &mut self,
        name: &str,
        engine: &mut VkEngine,
        init_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
        color_format: vk::Format,
        depth_format: Option<vk::Format>,
    ) -> Result<()> {
        let (clear_pass, load_pass) =
            engine.with_allocators(|ctx, res, alloc| {
                let clear_pass = res.create_render_pass_adv(
                    ctx,
                    color_format,
                    depth_format,
                    init_layout,
                    final_layout,
                    true,
                )?;

                let load_pass = res.create_render_pass_adv(
                    ctx,
                    color_format,
                    depth_format,
                    init_layout,
                    final_layout,
                    false,
                )?;

                VkEngine::set_debug_object_name(
                    ctx,
                    clear_pass,
                    &format!("{}-pass-clear", name),
                )?;
                VkEngine::set_debug_object_name(
                    ctx,
                    load_pass,
                    &format!("{}-pass-load", name),
                )?;

                let clear_pass_ix = res.insert_render_pass(clear_pass);
                let load_pass_ix = res.insert_render_pass(load_pass);

                Ok((clear_pass_ix, load_pass_ix))
            })?;

        let info = RenderPassInfo {
            clear_pass,
            load_pass,
            depth_format,
        };

        self.render_passes.insert(name.into(), info);

        Ok(())
    }
}
