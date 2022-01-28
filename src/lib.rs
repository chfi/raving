pub mod vk;

#[macro_export]
macro_rules! include_shader {
    ($file:expr) => {
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/", $file))
    };
}

#[macro_export]
macro_rules! load_shader {
    ($path:literal) => {{
        let buf = crate::include_shader!($path);
        let mut cursor = std::io::Cursor::new(buf);
        ash::util::read_spv(&mut cursor).unwrap()
    }};
}
