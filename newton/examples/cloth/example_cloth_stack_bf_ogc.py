###########################################################################
# Comparison test case: Stacked Cloth
#
# Dropping 50 layers of cloth on a fixed sphere.
#
# N: number of substeps per frame and iterations per substep (N^2
# iterations in total)
#
# difficulty: 'easy' or 'hard', only affecting the bending stiffness
#
###########################################################################

import math
import os

import numpy as np
import warp as wp
import warp.examples
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton import ParticleFlags
import time

import trimesh

N = 30
difficulty = 'easy'

class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = N  # must be an even number when using CUDA Graph
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = N
        # the BVH used by SolverVBD will be rebuilt every self.bvh_rebuild_frames
        # When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        # quality, in this case we need to completely rebuild the tree to achieve better query efficiency.
        self.bvh_rebuild_frames = 10

        # save a reference to the viewer
        self.viewer = viewer

        scene = newton.ModelBuilder()

        mesh = trimesh.load_mesh(newton.examples.get_asset("boundary.obj"), merge_norm=True, merge_tex=True)

        mesh_points = np.array(mesh.vertices)
        mesh_indices = np.array(mesh.faces).reshape((-1,))
        vertices = [wp.vec3(v) for v in mesh_points]

        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi/2),
            scale=0.3,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.15,
            tri_ke=30.0 if difficulty == 'easy' else 300.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0 if difficulty == 'easy' else 5e3,
            edge_kd=0.0,
        )

        fixed_point_indices = list(range(mesh_points.shape[0]))

        mesh = trimesh.load_mesh(newton.examples.get_asset("square.obj"), merge_norm=True, merge_tex=True)

        mesh_points = np.array(mesh.vertices)
        mesh_indices = np.array(mesh.faces).reshape((-1,))

        vertices = [wp.vec3(v) for v in mesh_points]

        for i in range(50):
            scene.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 1.0 + 0.1 * i),
                rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi/2),
                scale=1.0,
                vertices=vertices,
                indices=mesh_indices,
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.15,
                tri_ke=30.0 if difficulty == 'easy' else 300.0,
                tri_ka=0.0,
                tri_kd=0.0,
                edge_ke=0.0 if difficulty == 'easy' else 5e3,
                edge_kd=0.0
            )

        scene.color()
        self.model = scene.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.0

        self.faces = np.array(scene.tri_indices)

        if len(fixed_point_indices):
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in fixed_point_indices:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE

            self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=True,
            self_contact_radius=0.002,
            self_contact_margin=0.0035,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # put graph capture into it's own function
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return

        # Begin frame with time
        self.viewer.begin_frame(self.sim_time)

        # Render model-driven content (ground plane)
        self.viewer.log_state(self.state_0)

        self.viewer.end_frame()

    def test(self):
        p_lower = wp.vec3(-0.6, -0.9, -0.6)
        p_upper = wp.vec3(0.6, 0.9, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 1.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)

    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example, args)
