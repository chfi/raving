pub mod texture;
pub mod vk;

pub mod compositor;
pub mod scene;

pub mod script;

#[macro_export]
macro_rules! include_shader {
    ($file:expr) => {
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/", $file))
    };
}
