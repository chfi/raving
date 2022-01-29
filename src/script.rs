use crossbeam::atomic::AtomicCell;
use parking_lot::{Mutex, RwLock};

use lazy_static::lazy_static;

use rhai::plugin::*;

#[export_module]
pub mod vk {

    use ash::vk;

    #[rhai_mod(name = "ImageLayout")]
    pub mod image_layout {
        macro_rules! layout {
            ($l:ident) => {
                pub const $l: vk::ImageLayout = vk::ImageLayout::$l;
            };
        }

        layout!(UNDEFINED);
        layout!(PREINITIALIZED);
        layout!(GENERAL);
        layout!(TRANSFER_SRC_OPTIMAL);
        layout!(TRANSFER_DST_OPTIMAL);
        layout!(SHADER_READ_ONLY_OPTIMAL);
        layout!(COLOR_ATTACHMENT_OPTIMAL);
        layout!(PRESENT_SRC_KHR);
    }
}
