use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
};

use ash::{vk, Device};

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};
use rspirv_reflect::DescriptorInfo;
use thunderdome::Arena;

use super::*;

#[derive(Debug, Default, Clone)]
pub struct WindowResIndices {
    pub images: HashMap<String, ImageIx>,
    pub image_views: HashMap<String, ImageViewIx>,
    pub desc_sets: HashMap<
        String,
        HashMap<(vk::DescriptorType, vk::ImageLayout), DescSetIx>,
    >,
    pub framebuffers: HashMap<String, FramebufferIx>,
}

#[derive(Default)]
pub struct WindowResBuilder {
    pub images: HashMap<String, ImageRes>,
    pub image_views: HashMap<String, vk::ImageView>,
    pub desc_sets: HashMap<
        String,
        HashMap<(vk::DescriptorType, vk::ImageLayout), vk::DescriptorSet>,
    >,
    pub framebuffers: HashMap<String, vk::Framebuffer>,
}

impl WindowResBuilder {
    pub fn insert(
        self,
        index_map: &mut WindowResIndices,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> Result<()> {
        // clean up any existing resources first, in order
        for res_ix in index_map.framebuffers.values() {
            if let Some(fb) = res.framebuffers.remove(res_ix.0) {
                unsafe {
                    ctx.device().destroy_framebuffer(fb, None);
                }
            }
        }

        for res_ix in index_map.image_views.values() {
            if let Some(image_view) = res.image_views.remove(res_ix.0) {
                unsafe {
                    ctx.device().destroy_image_view(image_view, None);
                }
            }
        }

        for res_ix in index_map.images.values() {
            if let Some(image) = res.images.remove(res_ix.0) {
                res.free_image(ctx, alloc, image)?;
            }
        }

        // insert the new resources
        for (name, img) in self.images {
            if let Some(&ix) = index_map.images.get(&name) {
                // we already freed the resources above
                let _ = res.insert_image_at(ix, img);
            } else {
                let ix = res.insert_image(img);
                index_map.images.insert(name, ix);
            }
        }

        for (name, img_view) in self.image_views {
            if let Some(&ix) = index_map.image_views.get(&name) {
                let _ = res.insert_image_view_at(ix, img_view);
            } else {
                let ix = res.insert_image_view(img_view);
                index_map.image_views.insert(name, ix);
            }
        }

        for (name, inner) in self.desc_sets {
            for ((ty, layout), desc_set) in inner {
                if let Some(&ix) = index_map
                    .desc_sets
                    .get(&name)
                    .and_then(|i| i.get(&(ty, layout)))
                {
                    let _ = res.insert_desc_set_at(ix, desc_set);
                } else {
                    let ix = res.insert_desc_set(desc_set);
                    index_map
                        .desc_sets
                        .entry(name.to_string())
                        .or_default()
                        .insert((ty, layout), ix);
                }
            }
        }

        for (name, framebuffer) in self.framebuffers {
            if let Some(&ix) = index_map.framebuffers.get(&name) {
                let _ = res.insert_framebuffer_at(ix, framebuffer);
            } else {
                let ix = res.insert_framebuffer(framebuffer);
                index_map.framebuffers.insert(name, ix);
            }
        }

        Ok(())
    }
}

pub struct WindowResources {
    pub indices: WindowResIndices,

    desc_set_infos: HashMap<String, BTreeMap<u32, DescriptorInfo>>,

    desc_layout_infos: HashMap<String, DescriptorLayoutInfo>,

    // resources: Vec<WindowResDesc>,
    resources: HashMap<String, WindowResDesc>,
}

// pub struct WindowResDesc<Name: AsRef<str>> {
// name: Name,
#[derive(Clone)]
pub struct WindowResDesc {
    // pub name: String,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,

    descriptors: Vec<(vk::ImageUsageFlags, vk::ImageLayout)>,
    framebuffers: Vec<RenderPassIx>,
}

impl WindowResources {
    pub fn new() -> Self {
        let window_storage_set_info = make_descriptor_set_info(
            vk::DescriptorType::STORAGE_IMAGE,
            "out_image",
        );

        let window_texture_set_info = make_descriptor_set_info(
            vk::DescriptorType::SAMPLED_IMAGE,
            "out_image",
        );

        let window_storage_image_layout = make_descriptor_layout_info(
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
        );

        let window_texture_layout = make_descriptor_layout_info(
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT,
        );

        let desc_set_infos = [
            ("storage".to_string(), window_storage_set_info),
            ("sampled".to_string(), window_texture_set_info),
        ]
        .into_iter()
        .collect();

        let desc_layout_infos = [
            ("storage".to_string(), window_storage_image_layout),
            ("sampled".to_string(), window_texture_layout),
        ]
        .into_iter()
        .collect();

        Self {
            indices: Default::default(),

            desc_set_infos,
            desc_layout_infos,

            resources: Default::default(),
        }
    }

    pub fn add_image(
        &mut self,
        name: &str,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        descriptors: impl IntoIterator<
            Item = (vk::ImageUsageFlags, vk::ImageLayout),
        >,
        framebuffer_pass: Option<RenderPassIx>,
    ) -> Result<()> {
        let res = WindowResDesc {
            format,
            usage,
            // descriptors,
            descriptors: descriptors.into_iter().collect(),
            framebuffers: framebuffer_pass.into_iter().collect(),
        };

        self.resources.insert(name.to_string(), res);

        Ok(())
    }

    pub fn build(
        &self,
        engine: &mut VkEngine,
        width: u32,
        height: u32,
    ) -> Result<WindowResBuilder> {
        let mut builder = WindowResBuilder::default();

        engine.with_allocators(|ctx, res, alloc| {
            for (name, win_res) in self.resources.iter() {
                let image = res.allocate_image(
                    ctx,
                    alloc,
                    width,
                    height,
                    win_res.format,
                    win_res.usage,
                    Some(name),
                )?;

                let view = res.new_image_view(ctx, &image)?;

                for &pass_ix in &win_res.framebuffers {
                    let attchs = [view];
                    let framebuffer = res.create_framebuffer(
                        ctx, pass_ix, &attchs, width, height,
                    )?;

                    builder.framebuffers.insert(name.to_string(), framebuffer);
                }

                {
                    let desc_sets =
                        builder.desc_sets.entry(name.to_string()).or_default();

                    for &(usage, layout) in &win_res.descriptors {
                        let out = self
                            .allocate_desc_sets_for(res, view, usage, layout)?;

                        for (ty, set) in out {
                            desc_sets.insert((ty, layout), set);
                        }
                    }
                }

                builder.images.insert(name.to_string(), image);

                builder.image_views.insert(name.to_string(), view);
            }

            Ok(())
        })?;

        Ok(builder)
    }

    // pub fn add_image(&mut self,

    fn allocate_desc_sets_for(
        &self,
        res: &mut GpuResources,
        img_view: vk::ImageView,
        usage: vk::ImageUsageFlags,
        layout: vk::ImageLayout,
        // ) -> Result<Vec<(vk::ImageUsageFlags, vk::DescriptorSet)>> {
    ) -> Result<Vec<(vk::DescriptorType, vk::DescriptorSet)>> {
        let mut result = Vec::new();

        let window_storage_set_info =
            self.desc_set_infos.get("storage").unwrap();
        let window_sampled_set_info =
            self.desc_set_infos.get("sampled").unwrap();

        let window_storage_layout =
            self.desc_layout_infos.get("storage").unwrap();
        let window_sampled_layout =
            self.desc_layout_infos.get("sampled").unwrap();

        if usage.intersects(vk::ImageUsageFlags::SAMPLED) {
            let sampled_desc_set = res.allocate_desc_set_raw(
                window_sampled_layout,
                window_sampled_set_info,
                |res, builder| {
                    let info = ash::vk::DescriptorImageInfo::builder()
                        .image_layout(layout)
                        .image_view(img_view)
                        .build();

                    builder.bind_image(0, &[info]);

                    Ok(())
                },
            )?;

            result.push((vk::DescriptorType::SAMPLED_IMAGE, sampled_desc_set));
            // result.push((vk::ImageUsageFlags::SAMPLED, sampled_desc_set));
        }

        if usage.intersects(vk::ImageUsageFlags::STORAGE) {
            let storage_desc_set = res.allocate_desc_set_raw(
                window_storage_layout,
                window_storage_set_info,
                |res, builder| {
                    let info = ash::vk::DescriptorImageInfo::builder()
                        .image_layout(layout)
                        .image_view(img_view)
                        .build();

                    builder.bind_image(0, &[info]);

                    Ok(())
                },
            )?;

            result.push((vk::DescriptorType::STORAGE_IMAGE, storage_desc_set));
            // result.push((vk::ImageUsageFlags::STORAGE, storage_desc_set));
        }

        Ok(result)
    }
}

pub fn make_descriptor_layout_info(
    ty: vk::DescriptorType,
    stages: vk::ShaderStageFlags,
) -> DescriptorLayoutInfo {
    let mut info = DescriptorLayoutInfo::default();

    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(ty)
        .stage_flags(stages)
        .build();

    info.bindings.push(binding);
    info
}

pub fn make_descriptor_set_info(
    ty: vk::DescriptorType,
    name: &str,
) -> BTreeMap<u32, DescriptorInfo> {
    let info = DescriptorInfo {
        ty: rspirv_reflect::DescriptorType(ty.as_raw() as u32),
        binding_count: rspirv_reflect::BindingCount::One,
        name: name.to_string(),
    };
    Some((0u32, info)).into_iter().collect::<BTreeMap<_, _>>()
}

#[derive(Debug, Default, Clone)]
pub struct WinSizeIndices {
    pub images: HashMap<String, ImageIx>,
    pub image_views: HashMap<String, ImageViewIx>,
    pub desc_sets: HashMap<String, DescSetIx>,
    pub framebuffers: HashMap<String, FramebufferIx>,
}

#[derive(Default)]
pub struct WinSizeResourcesBuilder {
    pub images: HashMap<String, ImageRes>,
    pub image_views: HashMap<String, vk::ImageView>,
    pub desc_sets: HashMap<String, vk::DescriptorSet>,
    pub framebuffers: HashMap<String, vk::Framebuffer>,
}

impl WinSizeResourcesBuilder {
    pub fn insert(
        self,
        index_map: &mut WinSizeIndices,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> Result<()> {
        // clean up any existing resources first, in order
        for res_ix in index_map.framebuffers.values() {
            if let Some(fb) = res.framebuffers.remove(res_ix.0) {
                unsafe {
                    ctx.device().destroy_framebuffer(fb, None);
                }
            }
        }

        for res_ix in index_map.image_views.values() {
            if let Some(image_view) = res.image_views.remove(res_ix.0) {
                unsafe {
                    ctx.device().destroy_image_view(image_view, None);
                }
            }
        }

        for res_ix in index_map.images.values() {
            if let Some(image) = res.images.remove(res_ix.0) {
                res.free_image(ctx, alloc, image)?;
            }
        }

        // insert the new resources
        for (name, img) in self.images {
            if let Some(&ix) = index_map.images.get(&name) {
                // we already freed the resources above
                let _ = res.insert_image_at(ix, img);
            } else {
                let ix = res.insert_image(img);
                index_map.images.insert(name, ix);
            }
        }

        for (name, img_view) in self.image_views {
            if let Some(&ix) = index_map.image_views.get(&name) {
                let _ = res.insert_image_view_at(ix, img_view);
            } else {
                let ix = res.insert_image_view(img_view);
                index_map.image_views.insert(name, ix);
            }
        }

        for (name, desc_set) in self.desc_sets {
            if let Some(&ix) = index_map.desc_sets.get(&name) {
                let _ = res.insert_desc_set_at(ix, desc_set);
            } else {
                let ix = res.insert_desc_set(desc_set);
                index_map.desc_sets.insert(name, ix);
            }
        }

        for (name, framebuffer) in self.framebuffers {
            if let Some(&ix) = index_map.framebuffers.get(&name) {
                let _ = res.insert_framebuffer_at(ix, framebuffer);
            } else {
                let ix = res.insert_framebuffer(framebuffer);
                index_map.framebuffers.insert(name, ix);
            }
        }

        Ok(())
    }
}
