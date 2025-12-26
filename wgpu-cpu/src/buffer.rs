use std::{
    ops::{
        Bound,
        Deref,
        DerefMut,
        Range,
        RangeBounds,
    },
    sync::Arc,
};

use parking_lot::{
    ArcRwLockReadGuard,
    ArcRwLockWriteGuard,
    Mutex,
    RawRwLock,
    RwLock,
    RwLockReadGuard,
    RwLockWriteGuard,
};

#[derive(derive_more::Debug, Clone)]
pub struct Buffer {
    #[debug(skip)]
    data: Arc<RwLock<BufferData>>,
    #[debug(skip)]
    state: Arc<Mutex<BufferState>>,
    size: usize,
}

impl Buffer {
    pub fn new_unmapped(size: usize) -> Self {
        Self::new(size, None)
    }

    pub fn new_mapped_at_creation(size: usize) -> Self {
        Self::new(size, Some(wgpu::MapMode::Write))
    }

    pub fn new(size: usize, map_mode: Option<wgpu::MapMode>) -> Self {
        Self {
            data: Arc::new(RwLock::new(BufferData {
                data: vec![0; size],
            })),
            state: Arc::new(Mutex::new(BufferState {
                map_mode,
                map_request: None,
            })),
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    #[track_caller]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> BufferSlice {
        BufferSlice {
            buffer: self.clone(),
            range: sub_range(0..self.size, range),
        }
    }

    pub fn read(&self) -> BufferReadGuard<'_> {
        let state_guard = self.state.lock();
        state_guard.assert_unmapped();

        BufferReadGuard {
            data_guard: self.data.read(),
            state: self.state.clone(),
            range: 0..self.size,
        }
    }

    pub fn write(&self) -> BufferWriteGuard<'_> {
        let state_guard = self.state.lock();
        state_guard.assert_unmapped();

        BufferWriteGuard {
            data_guard: self.data.write(),
            state: self.state.clone(),
            range: 0..self.size,
        }
    }

    pub fn read_owned(&self) -> OwnedBufferReadGuard {
        let state_guard = self.state.lock();
        state_guard.assert_unmapped();

        OwnedBufferReadGuard {
            data_guard: self.data.read_arc(),
            state: self.state.clone(),
            range: 0..self.size,
        }
    }

    pub fn write_owned(&self) -> OwnedBufferWriteGuard {
        let state_guard = self.state.lock();
        state_guard.assert_unmapped();

        OwnedBufferWriteGuard {
            data_guard: self.data.write_arc(),
            state: self.state.clone(),
            range: 0..self.size,
        }
    }
}

impl wgpu::custom::BufferInterface for Buffer {
    fn map_async(
        &self,
        mode: wgpu::MapMode,
        range: Range<wgpu::BufferAddress>,
        callback: wgpu::custom::BufferMapCallback,
    ) {
        let mut state_guard = self.state.lock();

        if let Some(mapped) = state_guard.map_mode {
            panic!("Buffer already mapped: {mapped:?}");
        }
        else if let Some(map_request) = &state_guard.map_request {
            panic!("Buffer is already being mapped: {:?}", map_request.map_mode);
        }

        if let Some(data_guard) = self.data.try_write() {
            state_guard.map(mode);
        }
        else {
            state_guard.map_request = Some(BufferMapRequest {
                map_mode: mode,
                callback,
            });
        }
    }

    #[track_caller]
    fn get_mapped_range(
        &self,
        range: Range<wgpu::BufferAddress>,
    ) -> wgpu::custom::DispatchBufferMappedRange {
        let state_guard = self.state.lock();
        let range = sub_range(0..self.size, cast_range(range));

        let inner = match state_guard.map_mode {
            Some(wgpu::MapMode::Read) => {
                let data_guard = self.data.try_read_arc().expect("Range already mapped");
                BufferMappedRangeInner::Read(OwnedBufferReadGuard {
                    data_guard,
                    state: self.state.clone(),
                    range,
                })
            }
            Some(wgpu::MapMode::Write) => {
                let data_guard = self.data.try_write_arc().expect("Range already mapped");
                BufferMappedRangeInner::Write(OwnedBufferWriteGuard {
                    data_guard,
                    state: self.state.clone(),
                    range,
                })
            }
            None => panic!("Buffer is not mapped"),
        };

        wgpu::custom::DispatchBufferMappedRange::custom(BufferMappedRange { inner })
    }

    fn unmap(&self) {
        let mut state_guard = self.state.lock();
        if state_guard.map_mode.is_none() {
            panic!("Buffer is not mapped");
        }
        state_guard.unmap();
    }

    fn destroy(&self) {
        // nop
    }
}

#[derive(derive_more::Debug)]
struct BufferData {
    #[debug(skip)]
    data: Vec<u8>,
}

#[derive(Debug)]
struct BufferState {
    map_mode: Option<wgpu::MapMode>,
    map_request: Option<BufferMapRequest>,
}

impl BufferState {
    /// note: doesn't perform any checks
    fn map(&mut self, map_mode: wgpu::MapMode) {
        self.map_mode = Some(map_mode);
    }

    /// note: doesn't perform any checks
    fn unmap(&mut self) {
        self.map_mode = None;
    }

    #[track_caller]
    fn assert_map_mode(&self, map_mode: wgpu::MapMode) {
        match (self.map_mode, map_mode) {
            (None, _) => {
                panic!("Buffer is not mapped");
            }
            (Some(wgpu::MapMode::Read), wgpu::MapMode::Write) => {
                panic!("Buffer is only mapped for reading")
            }
            _ => {}
        }
    }

    #[track_caller]
    fn assert_unmapped(&self) {
        if let Some(map_mode) = self.map_mode {
            panic!("Buffer is mapped: {map_mode:?}");
        }
    }

    fn handle_map_request(&mut self) {
        if let Some(map_request) = self.map_request.take() {
            if let Some(mapped) = self.map_mode {
                panic!("Buffer already mapped: {mapped:?}");
            }

            self.map(map_request.map_mode);
            (map_request.callback)(Ok(()));
        }
    }
}

#[derive(derive_more::Debug)]
struct BufferMapRequest {
    map_mode: wgpu::MapMode,
    #[debug(skip)]
    callback: wgpu::custom::BufferMapCallback,
}

#[derive(derive_more::Debug)]
pub struct BufferReadGuard<'a> {
    #[debug(skip)]
    data_guard: RwLockReadGuard<'a, BufferData>,

    // we need this to call the callbacks for map requests
    state: Arc<Mutex<BufferState>>,

    range: Range<usize>,
}

impl<'a> BufferReadGuard<'a> {
    #[track_caller]
    pub fn slice(mut self, range: Range<usize>) -> Self {
        self.range = sub_range(self.range.clone(), range);
        self
    }
}

impl<'a> Deref for BufferReadGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data_guard.data
    }
}

impl<'a> Drop for BufferReadGuard<'a> {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.handle_map_request();
    }
}

#[derive(derive_more::Debug)]
pub struct BufferWriteGuard<'a> {
    #[debug(skip)]
    data_guard: RwLockWriteGuard<'a, BufferData>,

    // we need this to call the callbacks for map requests
    state: Arc<Mutex<BufferState>>,

    range: Range<usize>,
}

impl<'a> BufferWriteGuard<'a> {
    #[track_caller]
    pub fn slice(mut self, range: Range<usize>) -> Self {
        self.range = sub_range(self.range.clone(), range);
        self
    }
}

impl<'a> Deref for BufferWriteGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data_guard.data
    }
}

impl<'a> DerefMut for BufferWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data_guard.data
    }
}

impl<'a> Drop for BufferWriteGuard<'a> {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.handle_map_request();
    }
}

#[derive(derive_more::Debug)]
pub struct OwnedBufferReadGuard {
    #[debug(skip)]
    data_guard: ArcRwLockReadGuard<RawRwLock, BufferData>,

    // we need this to call the callbacks for map requests
    state: Arc<Mutex<BufferState>>,

    range: Range<usize>,
}

impl OwnedBufferReadGuard {
    #[track_caller]
    pub fn slice(mut self, range: Range<usize>) -> Self {
        self.range = sub_range(self.range.clone(), range);
        self
    }
}

impl<'a> Deref for OwnedBufferReadGuard {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data_guard.data
    }
}

impl Drop for OwnedBufferReadGuard {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.handle_map_request();
    }
}

#[derive(derive_more::Debug)]
pub struct OwnedBufferWriteGuard {
    #[debug(skip)]
    data_guard: ArcRwLockWriteGuard<RawRwLock, BufferData>,

    // we need this to call the callbacks for map requests
    state: Arc<Mutex<BufferState>>,

    range: Range<usize>,
}

impl OwnedBufferWriteGuard {
    #[track_caller]
    pub fn slice(mut self, range: Range<usize>) -> Self {
        self.range = sub_range(self.range.clone(), range);
        self
    }
}

impl<'a> Deref for OwnedBufferWriteGuard {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data_guard.data
    }
}

impl<'a> DerefMut for OwnedBufferWriteGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data_guard.data
    }
}

impl Drop for OwnedBufferWriteGuard {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.handle_map_request();
    }
}

#[derive(Clone, Debug)]
pub struct BufferSlice {
    pub buffer: Buffer,
    pub range: Range<usize>,
}

impl BufferSlice {
    pub fn from_wgpu_dispatch(
        buffer: &wgpu::custom::DispatchBuffer,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) -> Self {
        let buffer = buffer.as_custom::<Buffer>().unwrap().clone();
        let start = offset as usize;
        let end = start + size.map_or(buffer.size, |size| size.get() as usize);
        Self {
            buffer,
            range: start..end,
        }
    }

    pub fn read(&self) -> BufferReadGuard<'_> {
        self.buffer.read().slice(self.range.clone())
    }

    pub fn write(&self) -> BufferWriteGuard<'_> {
        self.buffer.write().slice(self.range.clone())
    }

    pub fn read_owned(&self) -> OwnedBufferReadGuard {
        self.buffer.read_owned().slice(self.range.clone())
    }

    pub fn write_owned(&self) -> OwnedBufferWriteGuard {
        self.buffer.write_owned().slice(self.range.clone())
    }
}

#[derive(Debug)]
pub struct BufferMappedRange {
    inner: BufferMappedRangeInner,
}

#[derive(Debug)]
enum BufferMappedRangeInner {
    Read(OwnedBufferReadGuard),
    Write(OwnedBufferWriteGuard),
}

impl wgpu::custom::BufferMappedRangeInterface for BufferMappedRange {
    fn slice(&self) -> &[u8] {
        match &self.inner {
            BufferMappedRangeInner::Read(guard) => &*guard,
            BufferMappedRangeInner::Write(guard) => &*guard,
        }
    }

    fn slice_mut(&mut self) -> &mut [u8] {
        match &mut self.inner {
            BufferMappedRangeInner::Read(guard) => panic!("Buffer is only mapped for reading"),
            BufferMappedRangeInner::Write(guard) => &mut *guard,
        }
    }
}

#[track_caller]
fn sub_range(range: Range<usize>, sub_range: impl RangeBounds<usize>) -> Range<usize> {
    let start = match sub_range.start_bound() {
        Bound::Included(index) => range.start + index,
        Bound::Excluded(index) => range.start + index + 1,
        Bound::Unbounded => range.start,
    };
    let end = match sub_range.end_bound() {
        Bound::Included(index) => range.start + index + 1,
        Bound::Excluded(index) => range.start + index,
        Bound::Unbounded => range.end,
    };

    assert!(
        start <= range.end,
        "New range start exceeds original range end: {start} < {}",
        range.end
    );

    assert!(
        end <= range.end,
        "New range end exceeds original range end: {end} < {}",
        range.end
    );

    start..end
}

fn cast_range(range: Range<wgpu::BufferAddress>) -> Range<usize> {
    (range.start as usize)..(range.end as usize)
}
