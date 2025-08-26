# SPDX-License-Identifier: Apache-2.0
# Plow through Sand (Implicit MPM) — fixed plate geometry, orientation, placement

import sys, math
import numpy as np
import warp as wp
wp.config.enable_backward = False

import newton
from newton.solvers import SolverImplicitMPM


# ------- kernels -------
@wp.kernel
def update_kinematic_mesh(
    rest_points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    R: wp.mat33,
    t: wp.vec3,
    dt: float,
):
    v = wp.tid()
    m = wp.mesh_get(mesh_id)
    p0 = m.points[v] + dt * m.velocities[v]
    rp = rest_points[v]
    np1 = wp.vec3(
        R[0,0]*rp[0] + R[0,1]*rp[1] + R[0,2]*rp[2] + t[0],
        R[1,0]*rp[0] + R[1,1]*rp[1] + R[1,2]*rp[2] + t[1],
        R[2,0]*rp[0] + R[2,1]*rp[1] + R[2,2]*rp[2] + t[2],
    )
    m.velocities[v] = (np1 - p0) / dt
    m.points[v]     = p0


@wp.kernel
def update_body_transform(
    body_q: wp.array(dtype=wp.transform),
    body_id: int,
    new_transform: wp.transform,
):
    tid = wp.tid()
    if tid == body_id:
        body_q[tid] = new_transform


# ------- helpers -------
def mat3_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s, C = math.cos(angle_rad), math.sin(angle_rad), 1.0 - math.cos(angle_rad)
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
    ], dtype=np.float32)


def make_box(center, size):
    cx, cy, cz = center
    sx, sy, sz = np.array(size, dtype=np.float32) * 0.5
    v = np.array([
        [cx - sx, cy - sy, cz - sz],  # 0
        [cx + sx, cy - sy, cz - sz],  # 1
        [cx - sx, cy + sy, cz - sz],  # 2
        [cx + sx, cy + sy, cz - sz],  # 3
        [cx - sx, cy - sy, cz + sz],  # 4
        [cx + sx, cy - sy, cz + sz],  # 5
        [cx - sx, cy + sy, cz + sz],  # 6
        [cx + sx, cy + sy, cz + sz],  # 7
    ], dtype=np.float32)
    t = np.array([
        [1,5,7],[1,7,3],   # +X
        [4,0,2],[4,2,6],   # -X
        [2,3,7],[2,7,6],   # +Y
        [4,5,1],[4,1,0],   # -Y
        [5,4,6],[5,6,7],   # +Z
        [0,1,3],[0,3,2],   # -Z
    ], dtype=np.int32)
    return v, t


def merge_meshes(parts):
    verts, tris, off = [], [], 0
    for v, t in parts:
        verts.append(v)
        tris.append(t + off)
        off += v.shape[0]
    V = np.vstack(verts).astype(np.float32)
    T = np.vstack(tris).astype(np.int32)
    return V, T


def spawn_particles(builder: newton.ModelBuilder, res, lo, hi, packing_fraction):
    Nx, Ny, Nz = [int(r) for r in res]
    px = np.linspace(lo[0], hi[0], Nx+1, dtype=np.float32)
    py = np.linspace(lo[1], hi[1], Ny+1, dtype=np.float32)
    pz = np.linspace(lo[2], hi[2], Nz+1, dtype=np.float32)
    points = np.stack(np.meshgrid(px, py, pz, indexing="ij")).reshape(3, -1).T

    cell = (hi - lo) / res
    radius = float(np.max(cell) * 0.5)
    volume = float(np.prod(cell)) * float(packing_fraction)

    # improved: use same randomization approach as granular example
    rng = np.random.default_rng()  # removed fixed seed for more natural variation
    points = points + 2.0 * radius * (rng.random(points.shape).astype(np.float32) - 0.5)
    vel = np.zeros_like(points, dtype=np.float32)

    builder.particle_q = points
    builder.particle_qd = vel
    builder.particle_mass  = np.full(points.shape[0], volume, dtype=np.float32)
    builder.particle_radius= np.full(points.shape[0], radius, dtype=np.float32)
    builder.particle_flags = np.zeros(points.shape[0], dtype=np.int32)
    print("Particle count:", points.shape[0])


class Example:
    def __init__(
        self,
        stage_path="example_sand_plow_fixed.usd",
        voxel_size=0.05,
        particles_per_cell=2.0,
        tolerance=1e-5,
        headless=False,
        sand_friction=0.55,
        dynamic_grid=True,
        plow_pitch_deg=-18.0,
        plow_speed=0.5,     # m/s along +X (left -> right)
    ):
        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, -9.81, 0.0)

        # timing - improved: use higher substeps like basic examples for better stability
        self.sim_time = 0.0
        self.frame_dt = 1.0/60.0
        self.sim_substeps = 5  # increased from 3, balanced for MPM performance
        self.sim_dt = self.frame_dt / self.sim_substeps

        # --- sand bed: much longer pile in X direction (plow travel direction) ---
        soil_w = 4.0  
        # FIXED: start sand bed slightly above ground plane to prevent glitching
        ground_clearance = voxel_size * 0.1  # half voxel size clearance
        lo = np.array([-soil_w/2, ground_clearance, -0.30], dtype=np.float32)
        hi = np.array([ soil_w/2, 0.18 + ground_clearance,  2.00], dtype=np.float32)
        res = np.array(np.ceil(particles_per_cell * (hi - lo) / voxel_size), dtype=int)
        spawn_particles(builder, res, lo, hi, packing_fraction=1.0)

        # --- MPM options (set before finalizing) ---
        opt = SolverImplicitMPM.Options()
        opt.voxel_size   = float(voxel_size)
        opt.max_fraction = 1.0
        opt.tolerance    = float(tolerance)
        opt.unilateral   = True  # improved: use unilateral like granular example
        opt.max_iterations = 250  # improved: use default from granular example
        opt.gauss_seidel = True   # improved: use Gauss-Seidel for better convergence
        opt.dynamic_grid = bool(dynamic_grid)
        if not dynamic_grid:
            opt.grid_padding = 5

        # improved: add yield stress configuration for better granular behavior
        opt.yield_stresses = (0.0, -1.0e8, 1.0e8)  # (yield, stretching, compression)

        # motion bounds (left->right along X) - for visual body init
        self.x0 = -soil_w/2.0  # start before the longer pile
        self.x1 = +soil_w/2.0  # end after the longer pile
        self.plow_y = 0.0
        self.plow_z = 1.0  # back to original Z position
        self.plow_speed = float(plow_speed)
        self.plow_finished = False  # track if plow has completed its pass

        # --- plow geometry ---
        plow_width    = 1.2        # span across Z
        bottom_len_x  = 0.50       # length along X (travel direction)
        bottom_thick  = 0.08       # thickness along Y (increased from 0.03 to prevent particle glitching)

        top_height_y  = 0.60       # vertical height
        top_thick_x   = 0.08       # thickness along X (increased from 0.03 to prevent particle glitching)

        # Bottom plate: a true plate in X–Z, small thickness in Y
        bottom_center = np.array([0.0, 0.14, 0.0], dtype=np.float32)
        bottom_size   = np.array([bottom_len_x, bottom_thick, plow_width], dtype=np.float32)

        # Top plate: vertical wall behind the bottom plate (negative X), slightly above it
        top_center    = np.array([-0.15, bottom_center[1] + 0.5*top_height_y + 0.07, 0.0], dtype=np.float32)
        top_size      = np.array([top_thick_x, top_height_y, plow_width], dtype=np.float32)

        v_bottom, t_bottom = make_box([0,0,0], bottom_size)
        v_top,    t_top    = make_box([0,0,0], top_size)

        # ==== FIX: pitch downward (front edge lower) and keep visual/collider identical ====
        pitch = math.radians(plow_pitch_deg)  # positive means "downward" now
        Rz = mat3_from_axis_angle(np.array([0,0,1], dtype=np.float32), pitch)

        # FIXED: Don't bake rotation into vertices, apply it in the kernel instead
        v_bottom = v_bottom + bottom_center
        v_top    = v_top + top_center

        V, T = merge_meshes([(v_bottom, t_bottom), (v_top, t_top)])
        indices_flat = T.reshape(-1).astype(np.int32)
        vels = np.zeros_like(V, dtype=np.float32)

        # Store the rotation matrix for use in the kernel
        self.plow_rotation = Rz

        self.plow_mesh = wp.Mesh(
            points=wp.array(V, dtype=wp.vec3),
            indices=wp.array(indices_flat, dtype=int),
            velocities=wp.array(vels, dtype=wp.vec3),
        )
        self.plow_rest = wp.array(V, dtype=wp.vec3)  # unrotated rest points

        # Visual body uses the EXACT same Rz (row-major) for the bottom plate
        def mat33_from_np_rowmajor(M: np.ndarray) -> wp.mat33:
            return wp.mat33(
                float(M[0,0]), float(M[0,1]), float(M[0,2]),
                float(M[1,0]), float(M[1,1]), float(M[1,2]),
                float(M[2,0]), float(M[2,1]), float(M[2,2]),
            )

        self.plow_body_id = builder.add_body(
            xform=wp.transform(wp.vec3(self.x0, self.plow_y, self.plow_z), wp.quat_identity())
        )

        builder.add_shape_box(
            self.plow_body_id,
            hx=bottom_size[0]*0.5,
            hy=bottom_size[1]*0.5,
            hz=bottom_size[2]*0.5,
            xform=wp.transform(
                wp.vec3(*bottom_center),
                wp.quat_from_matrix(mat33_from_np_rowmajor(Rz))   # <-- same Rz as collider
            ),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        builder.add_shape_box(
            self.plow_body_id,
            hx=top_size[0]*0.5,
            hy=top_size[1]*0.5,
            hz=top_size[2]*0.5,
            xform=wp.transform(
                wp.vec3(*top_center),
                wp.quat_from_matrix(mat33_from_np_rowmajor(Rz))   # FIXED: apply same rotation to top plate
            ),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        # ==== END FIX ====

        # finalize model & MPM
        self.model = builder.finalize()
        self.model.particle_mu = float(sand_friction)

        self.mpm = SolverImplicitMPM(self.model, opt)
        self.mpm.setup_collider(self.model, [self.plow_mesh])

        # improved: use dual state approach like granular example for stability
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.mpm.enrich_state(self.state_0)
        self.mpm.enrich_state(self.state_1)

        self.renderer = None if headless else newton.viewer.RendererOpenGL(self.model, stage_path)

        # Note: CUDA graphs not compatible with MPM solver operations
        self.use_cuda_graph = False
        self.graph = None

        # place plow at start
        self._update_plow_position(self.x0, self.plow_z, self.frame_dt)

    def _update_plow_position(self, x_pos: float, z_pos: float, dt: float):
        # FIXED: Apply both rotation and translation in the kernel
        R = self.plow_rotation  # use the stored rotation matrix
        t = np.array([x_pos, self.plow_y, z_pos], dtype=np.float32)

        # collider
        wp.launch(
            update_kinematic_mesh,
            dim=self.plow_rest.shape[0],
            inputs=[self.plow_rest, self.plow_mesh.id, wp.mat33(*R), wp.vec3(*t), float(dt)],
        )
        self.plow_mesh.refit()

        # visual
        new_tf = wp.transform(wp.vec3(x_pos, self.plow_y, z_pos), wp.quat_identity())
        if self.plow_body_id < self.model.body_count:
            wp.launch(
                update_body_transform,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.plow_body_id, new_tf],
            )

    def simulate_mpm(self):
        """MPM simulation loop - CUDA graphs not compatible with MPM operations."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.mpm.step(self.state_0, self.state_1, contacts=None, control=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not self.plow_finished:
            # single pass through the pile along X direction (left to right)
            x_current = self.x0 + self.plow_speed * self.sim_time

            if x_current >= self.x1:
                # plow has finished its pass
                x_current = self.x1
                self.plow_finished = True

            self._update_plow_position(x_current, self.plow_z, self.frame_dt)

        # run MPM simulation (CUDA graphs not compatible with MPM)
        self.simulate_mpm()

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state_0)  # improved: render current state
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage-path", type=lambda x: None if x == "None" else str(x),
                        default="example_sand_plow_fixed.usd")
    parser.add_argument("--num-frames", type=int, default=1200)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--sand-friction", "-mu", type=float, default=0.55)
    parser.add_argument("--tolerance", "-tol", type=float, default=1e-5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plow-pitch-deg", type=float, default=-18.0)
    parser.add_argument("--plow-speed", type=float, default=0.5)
    args = parser.parse_known_args()[0]

    if wp.get_device(args.device).is_cpu:
        print("Error: This example requires a GPU.")
        sys.exit(1)

    with wp.ScopedDevice(args.device):
        ex = Example(
            stage_path=args.stage_path,
            voxel_size=args.voxel_size,
            particles_per_cell=args.particles_per_cell,
            tolerance=args.tolerance,
            headless=args.headless,
            sand_friction=args.sand_friction,
            dynamic_grid=args.dynamic_grid,
            plow_pitch_deg=args.plow_pitch_deg,
            plow_speed=args.plow_speed,
        )
        for _ in range(args.num_frames):
            ex.step()
            ex.render()
        if ex.renderer:
            ex.renderer.save()