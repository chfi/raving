use crossbeam::atomic::AtomicCell;
use parking_lot::{Mutex, RwLock};

use lazy_static::lazy_static;

use rhai::plugin::*;

pub struct CVars {
    // scalars: RwLock<FxHashMap<String, AtomicCell<
    statics: RwLock<FxHashMap<String, Arc<rhai::Dynamic>>>,
}
