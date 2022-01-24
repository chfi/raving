use std::ffi::c_void;

use ash::{Device, Entry, Instance};

use ash::extensions::{ext::DebugUtils, khr::Surface};

use crossbeam::channel::{Receiver, Sender};

// use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};

use ash::vk::{KhrGetPhysicalDeviceProperties2Fn, StructureType};

use ash::vk;

pub struct GpuTask {
    cmd_bufs: Vec<vk::CommandBuffer>,

    wait: Vec<vk::Semaphore>,
    signal: Vec<vk::Semaphore>,
    // should also contain a reference to the command pool --
    // or some other way to identify the pool, so the buffers can be freed after use
    //
    // wait, is that necessary? i forget
}

// pub type GpuTask = Box<dyn FnOnce()

pub struct VkQueueThread {
    pub(super) queue: vk::Queue,
    pub queue_family_index: u32,
    // queue_index: u32,
    tasks_rx: Receiver<GpuTask>,
    tasks_tx: Sender<GpuTask>,

    present: bool,
    graphics: bool,
    compute: bool,
    transfer: bool,
    // present: Option<Receiver<GpuTask>>,
    // graphics: Option<Receiver<GpuTask>>,
    // compute: Option<Receiver<GpuTask>>,
    // transfer: Option<Receiver<GpuTask>>,
}

pub struct Queues {
    // queues: Vec<VkQueueThread>,
    pub thread: VkQueueThread,
    // task_tx: Sender<GpuTask>,
    // task_rx: Receiver<GpuTask>,
    // queue: VkQueueThread,
    /*
    graphics_tx: Sender<GpuTask>,
    graphics_rx: Receiver<GpuTask>,

    compute_tx: Sender<GpuTask>,
    compute_rx: Receiver<GpuTask>,

    transfer_tx: Sender<GpuTask>,
    transfer_rx: Receiver<GpuTask>,
    */
}

impl Queues {
    pub fn init(queue: vk::Queue, family_ix: u32) -> anyhow::Result<Self> {
        // 1st just find the graphics queue and use that for everything

        let (tasks_tx, tasks_rx) = crossbeam::channel::unbounded();

        let thread = VkQueueThread {
            queue_family_index: family_ix,
            queue,
            tasks_tx,
            tasks_rx,
            present: true,
            graphics: true,
            compute: true,
            transfer: true,
        };

        Ok(Queues { thread })
    }
}

pub struct VkContext {
    _entry: Entry,
    instance: Instance,

    debug_utils: Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>,

    surface: Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,

    #[allow(dead_code)]
    get_physical_device_features2: KhrGetPhysicalDeviceProperties2Fn,
}

impl VkContext {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    pub fn surface_khr(&self) -> vk::SurfaceKHR {
        self.surface_khr
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn debug_utils(&self) -> Option<&DebugUtils> {
        self.debug_utils.as_ref().map(|(utils, _)| utils)
    }
}

impl VkContext {
    pub fn new(
        entry: Entry,
        instance: Instance,
        debug_utils: Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>,
        surface: Surface,
        surface_khr: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        device: Device,
    ) -> anyhow::Result<Self> {
        let get_physical_device_features2 =
            unsafe {
                KhrGetPhysicalDeviceProperties2Fn::load(|name| {
                    std::mem::transmute(entry.get_instance_proc_addr(
                        instance.handle(),
                        name.as_ptr(),
                    ))
                })
            };

        Ok(VkContext {
            _entry: entry,
            instance,
            debug_utils,
            surface,
            surface_khr,
            physical_device,
            device,

            get_physical_device_features2,
        })
    }
}

impl VkContext {
    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    /// Find the first compatible format from `candidates`.
    pub fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                self.instance.get_physical_device_format_properties(
                    self.physical_device,
                    *candidate,
                )
            };
            (tiling == vk::ImageTiling::LINEAR
                && props.linear_tiling_features.contains(features))
                || (tiling == vk::ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }

    /// Return the maximum supported MSAA sample count.
    pub fn get_max_usable_sample_count(&self) -> vk::SampleCountFlags {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }
}

// impl Drop for VkContext {
//     fn drop(&mut self) {
//         unsafe {
//             self.device.destroy_device(None);
//             self.surface.destroy_surface(self.surface_khr, None);
//             if let Some((report, callback)) = self.debug_utils.take() {
//                 report.destroy_debug_utils_messenger(callback, None);
//             }
//             self.instance.destroy_instance(None);
//         }
//     }
// }
