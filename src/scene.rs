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

use nalgebra::{Similarity3, Transform3};
use nalgebra_glm as na;

pub mod index;

pub use index::*;

#[derive(Debug, Clone, Copy)]
struct RenderPassInfo {
    clear_pass: RenderPassIx,
    load_pass: RenderPassIx,
    depth_format: Option<vk::Format>,
}

#[derive(Debug, Clone, Copy)]
struct PipelineInfo {
    clear_pipeline: PipelineIx,
    load_pipeline: PipelineIx,
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

    transform: Similarity3<f32>,

    transform_uniform: BufferIx,
    vertices: BufferIx,

    sets: Vec<DescSetIx>,
}

#[derive(Debug, Clone)]
pub struct MaterialDef {
    pub id: MaterialId,
    pub name: rhai::ImmutableString,

    pass_info: RenderPassInfo,
    pipeline_info: PipelineInfo,

    sets: Vec<DescSetIx>,

    // TODO add some sort of attachment and desc set data
    vertex_desc: VertexDesc,
}

#[derive(Debug, Clone)]
pub struct VertexDesc {
    pub per_instance: bool,

    pub vertex_offset: usize,
    pub vertex_stride: usize,

    pub default_vertex_count: Option<usize>,
    pub default_instance_count: Option<usize>,

    pub attribute_formats: Vec<vk::Format>,
}

pub struct Scene {
    // cameras:
}

pub struct SceneCompositor {
    render_passes: HashMap<rhai::ImmutableString, RenderPassInfo>,

    materials: FxHashMap<MaterialId, MaterialDef>,
}

impl SceneCompositor {
    fn create_material(
        &self,
        engine: &mut VkEngine,
        id: MaterialId,
        vert_path: &str,
        frag_path: &str,
        fixed_sets: impl IntoIterator<Item = DescSetIx>,
    ) -> Result<MaterialDef> {
        let vert = engine
            .resources
            .load_shader(vert_path, vk::ShaderStageFlags::VERTEX)?;
        let frag = engine
            .resources
            .load_shader(frag_path, vk::ShaderStageFlags::FRAGMENT)?;

        let vert = engine.resources.insert_shader(vert);
        let frag = engine.resources.insert_shader(frag);

        let vertex_offset = 0;
        let vertex_stride = std::mem::size_of::<[f32; 3 * 3]>();

        let vert_binding_desc = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(vertex_stride as u32)
            .input_rate(vk::VertexInputRate::INSTANCE)
            .build();

        let pos_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        let norm_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12)
            .build();

        let color_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(24)
            .build();

        /*
        let uv_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12)
            .build();
        */

        let vertex_desc = VertexDesc {
            per_instance: false,

            vertex_offset,
            vertex_stride,

            default_vertex_count: None,
            default_instance_count: Some(1),

            attribute_formats: vec![vk::Format::R32G32B32_SFLOAT; 3],
        };

        let vert_binding_descs = [vert_binding_desc];
        let vert_attr_descs = [pos_desc, norm_desc, color_desc];

        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vert_binding_descs)
            .vertex_attribute_descriptions(&vert_attr_descs);

        let pass_info =
            *self.render_passes.get("scene-compositor-depth").unwrap();

        let (clear_pipeline, load_pipeline) =
            engine.with_allocators(|ctx, res, alloc| {
                let clear = res.create_graphics_pipeline(
                    ctx,
                    vert,
                    frag,
                    res[pass_info.clear_pass],
                    &vert_input_info,
                )?;

                let load = res.create_graphics_pipeline(
                    ctx,
                    vert,
                    frag,
                    res[pass_info.load_pass],
                    &vert_input_info,
                )?;

                Ok((clear, load))
            })?;

        let pipeline_info = PipelineInfo {
            clear_pipeline,
            load_pipeline,
        };

        let material = MaterialDef {
            id,
            name: "test-material".into(),

            pass_info,
            pipeline_info,

            sets: fixed_sets.into_iter().collect(),

            vertex_desc,
        };

        Ok(material)
    }

    pub fn init(
        engine: &mut VkEngine,
        init_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Result<Self> {
        let mut scomp = Self {
            render_passes: HashMap::default(),
            materials: FxHashMap::default(),
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
