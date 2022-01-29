use ash::vk;
use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use parking_lot::{Mutex, RwLock};

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use std::sync::Arc;

use crate::vk::{
    context::VkContext, resource::index::*, util::LineRenderer, GpuResources,
};

use rhai::plugin::*;

use super::vk as rvk;

pub type WithAllocators<T> = Box<
    dyn FnOnce(
            &VkContext,
            &mut GpuResources,
            &mut Allocator,
        ) -> anyhow::Result<T>
        + Send
        + Sync,
>;

// pub struct VariableMap<T> {
//     map: FxHashMap<usize
// }

pub enum Resolver<T> {
    Resolver(Option<WithAllocators<T>>),
    Value(T),
}

impl<T> Resolver<T> {
    pub fn resolve(
        &mut self,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> Option<&T> {
        match self {
            Resolver::Resolver(resolver) => {
                let f = resolver.take()?;

                let v = f(ctx, res, alloc).ok()?;

                *self = Self::Value(v);

                self.get()
            }
            Resolver::Value(v) => Some(v),
        }
    }

    pub fn get(&self) -> Option<&T> {
        if let Self::Value(v) = &self {
            Some(v)
        } else {
            None
        }
    }

    pub fn can_resolve(&self) -> bool {
        !matches!(self, Self::Resolver(None))
    }

    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Value(_))
    }
}

impl<T> std::default::Default for Resolver<T> {
    fn default() -> Self {
        Self::Resolver(None)
    }
}

pub type VariableMap<T> = Arc<Mutex<FxHashMap<usize, Mutex<Resolver<T>>>>>;
// Arc<Mutex<FxHashMap<usize, Mutex<Option<WithAllocators<T>>>>>>;

#[derive(Default, Clone)]
pub struct BatchBuilder {
    image_vars: VariableMap<ImageIx>,
    image_view_vars: Arc<Mutex<FxHashMap<usize, usize>>>,

    buffer_vars: VariableMap<BufferIx>,

    pipeline_vars: VariableMap<PipelineIx>,

    desc_set_vars: VariableMap<DescSetIx>,
}

impl BatchBuilder {
    pub fn allocate_image(
        &self,
        w: u32,
        h: u32,
        fmt: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> usize {
        let resolver = Box::new(
            move |ctx: &VkContext,
                  res: &mut GpuResources,
                  alloc: &mut Allocator| {
                res.allocate_image(ctx, alloc, w, h, fmt, usage, None)
            },
        ) as WithAllocators<ImageIx>;
        self.new_image_resolver(resolver)
    }

    pub fn new_image_resolver(
        &self,
        resolver: WithAllocators<ImageIx>,
    ) -> usize {
        let mut vars = self.image_vars.lock();

        let i = vars.len();

        vars.insert(i, Mutex::new(Resolver::Resolver(Some(resolver))));

        i
    }

    pub fn new_image_view(&self, img: usize) -> Option<usize> {
        let mut vars = self.image_view_vars.lock();
        let i = vars.len();
        vars.insert(i, img);
        Some(i)
    }

    pub fn get_image(&self, img: usize) -> Option<ImageIx> {
        let mut vars = self.image_vars.lock();
        let resolver = vars.get_mut(&img)?;
        let r = resolver.get_mut();
        let v = r.get()?;
        Some(*v)
    }

    pub fn resolve_image(
        &self,
        img: usize,
        ctx: &VkContext,
        res: &mut GpuResources,
        alloc: &mut Allocator,
    ) -> Option<ImageIx> {
        let mut vars = self.image_vars.lock();
        let resolver = vars.get_mut(&img)?;
        let r = resolver.get_mut();
        let v = r.resolve(ctx, res, alloc)?;
        Some(*v)
    }

    pub fn create_engine(&self) -> rhai::Engine {
        let mut engine = rhai::Engine::new();

        let self_arc = Arc::new(self.clone());
        let other = self_arc.clone();

        engine.register_result_fn(
            "allocate_image",
            move |width: i64,
                  height: i64,
                  fmt: ash::vk::Format,
                  usage: ash::vk::ImageUsageFlags| {
                let img = other.allocate_image(
                    width as u32,
                    height as u32,
                    fmt,
                    usage,
                );

                Ok(img)
            },
        );

        let other = self_arc.clone();

        engine.register_result_fn(
            "create_image_view",
            move |image_var: usize| {
                let i = other.new_image_view(image_var).unwrap();
                Ok(i)
            },
        );

        todo!();
    }
}

pub struct Console {
    // input_history: Vec<String>,
    output_log: Vec<String>,

    renderer: LineRenderer,
}

impl Console {
    pub fn create_engine() -> rhai::Engine {
        let mut engine = rhai::Engine::new();

        todo!();
    }
}
