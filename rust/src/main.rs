// who thought it would be a good idea to base all the quadtree
// code for f32????!?!?!?!?!?

use std::cell::RefCell;

use quadtree::{BHQuadtree, Vec2, WeightedPoint};
use rand::Rng;
use rayon::prelude::*;
use three_d::*;
use wide::f32x4;

// some wide-related functions
// uses more modern cpu calls to process stuff in batches
// i will NOT be using the gpu in ANY scenario (CUDA PTSD)
fn accumulate_batch(target: Vec2, points: &[WeightedPoint]) -> Vec2 {
    let mut acc_x = 0.0;
    let mut acc_y = 0.0;

    for chunk in points.chunks_exact(4) {
        let px = f32x4::from([chunk[0].pos.x, chunk[1].pos.x, chunk[2].pos.x, chunk[3].pos.x]);
        let py = f32x4::from([chunk[0].pos.y, chunk[1].pos.y, chunk[2].pos.y, chunk[3].pos.y]);
        let m  = f32x4::from([chunk[0].mass,  chunk[1].mass,  chunk[2].mass,  chunk[3].mass]);

        let tx = f32x4::splat(target.x);
        let ty = f32x4::splat(target.y);

        let dx = px - tx;
        let dy = py - ty;

        let dist_sq = dx * dx + dy * dy + f32x4::splat(0.5);

        let inv_dist_sq = f32x4::ONE / dist_sq;
        let inv_dist = inv_dist_sq.sqrt();
        let strength = m * inv_dist_sq;

        let fx = dx * strength * inv_dist;
        let fy = dy * strength * inv_dist;

        acc_x += fx.reduce_add();
        acc_y += fy.reduce_add();
    }

    for wp in points.chunks_exact(4).remainder() {
        let dir = wp.pos - target;
        let dist_sq = dir.length_squared() + 0.5;
        let inv_dist_sq = 1.0 / dist_sq;
        let inv_dist = inv_dist_sq.sqrt();
        let strength = wp.mass * inv_dist_sq;
        acc_x += dir.x * strength * inv_dist;
        acc_y += dir.y * strength * inv_dist;
    }

    Vec2::new(acc_x, acc_y)
}

fn accumulate_simd(quad: &quadtree::BHQuadtree, target: Vec2) -> Vec2 {
    let buffer = RefCell::new(Vec::with_capacity(32));
    let total = RefCell::new(Vec2::ZERO);

    quad.accumulate(target, |wp: WeightedPoint| {
        let mut buf = buffer.borrow_mut();
        buf.push(wp);

        if buf.len() >= 4 {
            let force = accumulate_batch(target, &buf);
            *total.borrow_mut() += force;
            buf.clear();
        }

        Vec2::ZERO // must return something, but ignored
    });

    let buf = buffer.into_inner();
    if !buf.is_empty() {
        *total.borrow_mut() += accumulate_batch(target, &buf);
    }

    total.into_inner()
}

#[derive(Clone)]
struct Particle {
    pub pos: Vec2,
    pub vel: Vec2,
    pub mass: f32,
    //pub color: [u8; 4],
    //pub size: f32,
}

impl Particle {
    pub fn new(x: f32, y: f32) -> Self {
        Particle {
            pos: Vec2::new(x, y),
            vel: Vec2::new(0., 0.),
            mass: 5., //color: [255, 0, 0, 255],
                      //size: 5.,
        }
    }
}

impl quadtree::Point for Particle {
    fn point(&self) -> Vec2 {
        self.pos
    }
}

impl From<Particle> for WeightedPoint {
    fn from(value: Particle) -> Self {
        WeightedPoint {
            pos: value.pos,
            mass: value.mass,
        }
    }
}

struct Simulation {
    bh_quad: BHQuadtree,
    particles: Vec<Particle>,
    weighted_buff: Vec<WeightedPoint>,
    bound_min: Vec2,
    bound_max: Vec2,
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            bh_quad: BHQuadtree::new(0.85),
            particles: Vec::new(),
            weighted_buff: Vec::new(),
            bound_min: Vec2::ZERO,
            bound_max: Vec2::ZERO,
        }
    }

    pub fn add_particle(&mut self, particle: Particle) {
        self.particles.push(particle);

        if self.weighted_buff.capacity() < self.particles.len() {
            self.weighted_buff
                .reserve(self.particles.len() - self.weighted_buff.capacity());
        }
    }

    fn update_bounds(&mut self) {
        if self.particles.is_empty() {
            return;
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for p in &self.particles {
            min_x = min_x.min(p.pos.x);
            max_x = max_x.max(p.pos.x);
            min_y = min_y.min(p.pos.y);
            max_y = max_y.max(p.pos.y);
        }

        const INTERACTION_RADIUS: f32 = 158.0; // sqrt(25000) ≈ 158
        self.bound_min = Vec2::new(min_x - INTERACTION_RADIUS, min_y - INTERACTION_RADIUS);
        self.bound_max = Vec2::new(max_x + INTERACTION_RADIUS, max_y + INTERACTION_RADIUS);
    }

    // hah thanks galaxy engine (theyre nice people)
    pub fn add_particle_galaxy(&mut self, center: Vec2, radius: f32, particle: Particle) {
        let mut p = particle;

        let center_dir = p.pos - center;
        let distance = center_dir.length();

        if distance > 0.0 {
            let orbital_speed = (50.0 / distance.max(10.0)).sqrt() * 0.7;
            p.vel = Vec2::new(-center_dir.y, center_dir.x).normalize() * orbital_speed;
        }

        self.particles.push(p);

        if self.weighted_buff.capacity() < self.particles.len() {
            self.weighted_buff
                .reserve(self.particles.len() - self.weighted_buff.capacity());
        }
    }

    fn update(&mut self) {
        self.update_bounds();

        self.weighted_buff.clear();
        self.weighted_buff
            .par_extend(self.particles.par_iter().map(|p| WeightedPoint {
                pos: p.pos,
                mass: p.mass,
            }));

        self.bh_quad
            .build(std::mem::take(&mut self.weighted_buff), 24);

        self.particles.par_iter_mut().for_each(|p| {
            // ignore particles certain distance away
            //if p.pos.x < self.bound_min.x || p.pos.x > self.bound_max.x ||
            //   p.pos.y < self.bound_min.y || p.pos.y > self.bound_max.y {
            //    return;
            //}
            let target = p.pos;

            /*let total_force: Vec2 = self.bh_quad.accumulate(target, |wp: WeightedPoint| {
                if wp.pos == target {
                    return Vec2::ZERO;
                }

                let dir = wp.pos - target;
                let mut dist_sq = dir.length_squared() + 0.5;

                //if dist_sq < f32::MIN_POSITIVE {
                //    dist_sq = f32::MIN_POSITIVE;
                //}

                //if dist_sq > 25000. {
                //    return Vec2::ZERO;
                //}

                let inv_dist_sq = 1.0 / dist_sq;
                let inv_dist = inv_dist_sq.sqrt();
                let strength = wp.mass * inv_dist_sq;

                dir * strength * inv_dist
            });*/

            let total_force = accumulate_simd(&self.bh_quad, target);

            let accel = 50. * (total_force / p.mass);

            let dt = (0.1 / accel.length().max(0.8)).min(0.2);
            p.pos += p.vel * dt + 0.5 * accel * dt * dt;
            p.vel += accel * dt;
        });
    }
}

fn main() {
    let (window_width, window_height) = (1280., 720.);
    let window = Window::new(WindowSettings {
        title: "barnes hutt particle simulation".to_string(),
        min_size: (window_width as u32, window_height as u32),
        ..Default::default()
    })
    .unwrap();
    let (context, window_viewport, window_dpi) =
        (window.gl(), window.viewport(), window.device_pixel_ratio());
    let mut rng = rand::rng();
    let mut camera = Camera::new_2d(window_viewport);
    let mut control = Control2D::new(0.5, 500.);
    let mut simulation = Simulation::new();

    for _ in 0..50000 {
        let particle = Particle::new(
            rng.random_range(-3000.0..window_width) + window_width / 2.,
            rng.random_range(-3000.0..window_height) + window_height / 2.,
        );
        simulation.add_particle(particle);
    }

    let particle_cpu_mat = CpuMaterial {
        albedo: Srgba::RED,
        ..Default::default()
    };
    let mut particle_mesh = Gm::new(
        InstancedMesh::new(&context, &Instances::default(), &CpuMesh::circle(8)),
        ColorMaterial::new(&context, &particle_cpu_mat),
    );

    //bh_quad.build(particles.iter().cloned().map(Into::into).collect(), 4);

    window.render_loop(move |mut frame_input| {
        camera.set_viewport(frame_input.viewport);
        control.handle_events(&mut camera, &mut frame_input.events, window_dpi);
        //update_simulation(&mut bh_quad, &mut particles);

        simulation.update();

        /*for (mesh, p) in particle_meshes.iter_mut().zip(particles.iter()) {
            mesh.set_transformation(Mat4::from_translation(vec3(
                p.pos.x - window_width / 2.0,
                p.pos.y - window_height / 2.0,
                0.0,
            )));
        }*/

        let view_matrix = camera.view();
        let projection_matrix = camera.projection();
        let view_projection = projection_matrix * view_matrix;

        // i hate life
        let transforms: Vec<Mat4> = simulation
            .particles
            .par_iter()
            .filter(|p| {
                let world_pos = vec3(
                    p.pos.x - window_width / 2.0,
                    p.pos.y - window_height / 2.0,
                    0.0,
                );

                let clip_pos = view_projection * vec4(world_pos.x, world_pos.y, world_pos.z, 1.0);

                if clip_pos.w != 0.0 {
                    let ndc_x = clip_pos.x / clip_pos.w;
                    let ndc_y = clip_pos.y / clip_pos.w;

                    ndc_x >= -1.1 && ndc_x <= 1.1 && ndc_y >= -1.1 && ndc_y <= 1.1
                } else {
                    false
                }
            })
            .map(|p| {
                Mat4::from_translation(vec3(
                    p.pos.x - window_width / 2.0,
                    p.pos.y - window_height / 2.0,
                    0.0,
                )) * Mat4::from_scale(5.0)
            })
            .collect();

        particle_mesh.set_instances(&Instances {
            transformations: transforms,
            ..Default::default()
        });

        frame_input
            .screen()
            .clear(ClearState::color_and_depth(0., 0., 0., 1., 1.))
            .render(&camera, &mut particle_mesh.into_iter(), &[]);

        FrameOutput::default()
    });
}
