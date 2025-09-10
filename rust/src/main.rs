// who thought it would be a good idea to base all the quadtree
// code for f32????!?!?!?!?!?

use quadtree::{BHQuadtree, Vec2, WeightedPoint};
use rand::Rng;
use rayon::prelude::*;
use three_d::*;

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
            bh_quad: BHQuadtree::new(1.5),
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
            .extend(self.particles.iter().map(|p| WeightedPoint {
                pos: p.pos,
                mass: p.mass,
            }));

        self.bh_quad
            .build(std::mem::take(&mut self.weighted_buff), 6);

        self.particles.par_iter_mut().for_each(|p| {
            // ignore particles certain distance away
            //if p.pos.x < self.bound_min.x || p.pos.x > self.bound_max.x ||
            //   p.pos.y < self.bound_min.y || p.pos.y > self.bound_max.y {
            //    return;
            //}
            let target = p.pos;

            let total_force: Vec2 = self.bh_quad.accumulate(target, |wp: WeightedPoint| {
                if wp.pos == target {
                    return Vec2::ZERO;
                }

                let dir = wp.pos - target;
                let dist_sq = dir.length_squared() + 20.;

                if dist_sq > 25000. {
                    return Vec2::ZERO;
                }

                let inv_dist_sq = 1.0 / dist_sq;
                let inv_dist = inv_dist_sq.sqrt();
                let strength = wp.mass * inv_dist_sq;

                dir * strength * inv_dist
            });

            let accel = 50. * total_force / p.mass;

            let dt = (0.25 / accel.length().max(0.1)).min(0.5);
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
        InstancedMesh::new(&context, &Instances::default(), &CpuMesh::circle(3)),
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
            .render(&camera, std::iter::once(&mut particle_mesh), &[]);

        FrameOutput::default()
    });
}
