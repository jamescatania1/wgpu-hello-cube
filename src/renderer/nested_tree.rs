use std::collections::HashMap;

pub struct TwoTree {
    pub size: glam::UVec3,
    pub chunks: Vec<Chunk>,
    pub data: Vec<u8>,

    /// Maps chunk sizes (2, 4, 8, 16, ..., 512) to free start indices in the `data` vector
    free_list: HashMap<u32, Vec<u32>>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Chunk {
    brick_index: u32,
    brick_size: u32,
    mask: [u32; 16],
}
impl Default for Chunk {
    fn default() -> Self {
        Self {
            brick_index: 0,
            brick_size: 0,
            mask: [0; 16],
        }
    }
}

// #[repr(C)]
// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// pub struct Brick {
//     data: Vec<u32>,
// }
// impl Default for Brick {
//     fn default() -> Self {
//         Self { data: vec![0; 2] }
//     }
// }

fn chunk_index(size: glam::UVec3, pos: glam::UVec3) -> u32 {
    (pos.x >> 3) * size.y * size.z + (pos.y >> 3) * size.z + (pos.z >> 3)
}
fn voxel_index(pos: glam::UVec3) -> u32 {
    (pos.x & 7) << 6 | (pos.y & 7) << 3 | (pos.z & 7)
}

impl TwoTree {
    // pub const fn bitmask_lut() -> [u64; 64 * 8] {
    //     // for ray_dir in 0..8 {
    //     //     let dir = glam::
    //     // }
    // }
    //

    pub fn new(size: glam::UVec3) -> Self {
        let size = size.map(|x| (x + 7) >> 3);
        Self {
            size,
            chunks: vec![Chunk::default(); size.element_product() as usize],
            free_list: HashMap::new(),
            data: Vec::new(),
        }
    }

    pub fn get(&self, pos: glam::UVec3) -> u8 {
        let chunk = &self.chunks[chunk_index(self.size, pos) as usize];

        let targ = voxel_index(pos);
        if chunk.mask[(targ >> 5) as usize] & (1 << (targ & 31)) == 0 {
            return 0;
        }

        let mut cur = chunk.brick_index;
        for i in 0..targ {
            if chunk.mask[(i >> 5) as usize] & (1 << (i & 31)) != 0 {
                cur += 1
            }
        }
        self.data[cur as usize]
    }

    pub fn insert(&mut self, pos: glam::UVec3, value: u8) {
        let chunk = &mut self.chunks[chunk_index(self.size, pos) as usize];
        let targ = (pos.x & 7) << 6 | (pos.y & 7) << 3 | (pos.z & 7);

        let brick_count = chunk.mask.iter().map(|m| m.count_ones()).sum::<u32>();
        let mut target_count = brick_count;

        let existing = chunk.mask[(targ >> 5) as usize] & (1 << (targ & 31)) != 0;
        if value == 0 {
            if existing {
                target_count -= 1;
            }
            chunk.mask[(targ >> 5) as usize] &= !(1 << (targ & 31));
        } else {
            if !existing {
                target_count += 1;
            }
            chunk.mask[(targ >> 5) as usize] |= 1 << (targ & 31);
        }

        if chunk.brick_size == 0 {
            if value == 0 {
                return;
            }

            // allocate new brick (start at size 4)
            chunk.brick_index = self
                .free_list
                .get_mut(&4)
                .and_then(|list| list.pop())
                .unwrap_or_else(|| {
                    let start_idx = self.data.len() as u32;
                    self.data.resize(self.data.len() + 4, 0);
                    start_idx
                });
            chunk.brick_size = 4;
        }

        if target_count >= chunk.brick_size {
            // grow brick by factor of two
            let new_idx = self
                .free_list
                .get_mut(&(chunk.brick_size << 1))
                .and_then(|list| list.pop())
                .unwrap_or_else(|| {
                    let start_idx = self.data.len() as u32;
                    self.data
                        .resize(self.data.len() + (chunk.brick_size << 1) as usize, 0);
                    start_idx
                });
            self.data.copy_within(
                (chunk.brick_index as usize)..((chunk.brick_index + chunk.brick_size) as usize),
                new_idx as usize,
            );
            // also, we zero out old brick and add it to the free list
            self.data
                [(chunk.brick_index as usize)..((chunk.brick_index + chunk.brick_size) as usize)]
                .fill(0);
            if let Some(list) = self.free_list.get_mut(&chunk.brick_size) {
                list.push(chunk.brick_index);
            } else {
                self.free_list
                    .insert(chunk.brick_size, vec![chunk.brick_index]);
            }

            chunk.brick_index = new_idx;
            chunk.brick_size <<= 1;
        }

        // packed offset of the voxel in the current brick
        let mut voxel_offset = 0;
        for i in 0..targ {
            if chunk.mask[(i >> 5) as usize] & (1 << (i & 31)) != 0 {
                voxel_offset += 1
            }
        }

        if existing && value == 0 {
            for i in voxel_offset..(brick_count - 1) {
                self.data[(chunk.brick_index + i) as usize] =
                    self.data[(chunk.brick_index + i + 1) as usize];
            }
            self.data[(chunk.brick_index + brick_count - 1) as usize] = 0;
        } else if !existing && value != 0 {
            for i in (voxel_offset..brick_count).rev() {
                self.data[(chunk.brick_index + i + 1) as usize] =
                    self.data[(chunk.brick_index + i) as usize];
            }
            self.data[(chunk.brick_index + voxel_offset) as usize] = value;
        } else {
            self.data[(chunk.brick_index + voxel_offset) as usize] = value;
        }
    }

    pub fn from_scene(scene: &crate::vox::Scene) -> Self {
        let timer = std::time::Instant::now();

        let x_run = scene.size.y as usize * scene.size.z as usize;
        let y_run = scene.size.z as usize;

        let mut _self = Self::new(scene.size.as_uvec3());
        let mut voxels = vec![0; scene.size.element_product() as usize];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                let index = pos.x * x_run + pos.y * y_run + pos.z;
                voxels[index] = palette_index;
            }
        }

        for x in 0..scene.size.x {
            for y in 0..scene.size.y {
                for z in 0..scene.size.z {
                    let pos = glam::ivec3(x, y, z).as_uvec3();
                    let index = pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                    let voxel = voxels[index];
                    _self.insert(pos, voxel);
                }
            }
        }
        // let _t = std::time::Instant::now();
        // _self.merge_optimize();
        // println!("merge optmization took {:#?}", _t.elapsed());

        {
            let mut errors = 0;
            for x in 0..scene.size.x {
                for y in 0..scene.size.y {
                    for z in 0..scene.size.z {
                        let pos = glam::ivec3(x, y, z).as_uvec3();
                        let index =
                            pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                        let voxel = voxels[index];
                        // assert_eq!(voxel, tree.get(pos));
                        let tval = _self.get(pos);
                        if tval != voxel {
                            println!(
                                "pos {} failed. actual: {}, tree: {}",
                                pos,
                                voxel,
                                _self.get(pos)
                            );
                            errors += 1;
                        }
                    }
                }
            }
            println!("finished with {} errors", errors);
        }

        println!("built tree in {:#?}", timer.elapsed());
        println!(
            "tree length: {} mb",
            (_self.chunks.len() * 72 + _self.data.len()) as f64 / 1000000.0
        );
        dbg!(&_self.free_list);
        for i in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
            println!(
                "chunks size {}: {}",
                i,
                &_self
                    .chunks
                    .iter()
                    .map(|c| if c.brick_size == i { 1 } else { 0 })
                    .sum::<i64>()
            );
        }
        println!(
            "original length: {} mb",
            (scene.size.element_product()) as f64 / 1000000.0
        );
        _self
    }
}
