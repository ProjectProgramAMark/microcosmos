import functools
import subprocess
import time

import jax
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

import jax.numpy as jnp
import numpy as np
from PIL import Image

from microcosmos.utils import jax_timer

BASE_COLOR = jnp.array([0.3, 0.6, 0.9])
COLOR_VARIATION = jnp.array([0.3, 0.2, 0.0])

@functools.partial(jax.jit, static_argnames=["sz", "render_particles", "uniform_color"])
def render(node_timeseries, field_timeseries, sz, render_particles=True, uniform_color=False) -> jax.Array:
    frames = node_timeseries.position.shape[0]
    H = field_timeseries.grid_shape[0]
    W = field_timeseries.grid_shape[1]

    node_color = node_timeseries.color_bend[0]

    # Create base texture from potentials
    render_tex = 1 - jnp.clip(field_timeseries.steric, 0, 1)
    render_tex = jnp.stack([render_tex] * 3, axis=-1)

    if render_particles:
        # Convert positions to clipped pixel coordinates
        pixel_pos = jnp.floor(node_timeseries.position).astype(int)

        pixel_x = jnp.clip(pixel_pos[..., 0], 0, W - 1)
        pixel_y = jnp.clip(pixel_pos[..., 1], 0, H - 1)

        # Generate particle colors and frame indices
        num_particles = pixel_pos.shape[1]
        frame_indices = jnp.arange(frames)[:, None]
        indices = (
            frame_indices.repeat(num_particles, axis=1), # (frames, num_particles)
            pixel_y,                                    # (frames, num_particles)
            pixel_x                                     # (frames, num_particles)
        )

        # Create colors based on graph coloring or uniform blue
        if uniform_color:
            particle_colors = jnp.broadcast_to(BASE_COLOR, (num_particles, 3))
        else:
            particle_colors = BASE_COLOR + COLOR_VARIATION * node_color[:, None]
        colors = jnp.broadcast_to(particle_colors[None, :, :], (frames, num_particles, 3))

        # Set particle colors
        render_tex = render_tex.at[indices].set(colors)

        # Render debug vectors
        # Calculate absolute position: Node Position + Debug Displacement
        debug_pos = jnp.floor(node_timeseries.position + node_timeseries.debug_vector).astype(int)

        debug_x = jnp.clip(debug_pos[..., 0], 0, W - 1)
        debug_y = jnp.clip(debug_pos[..., 1], 0, H - 1)

        debug_indices = (
            frame_indices.repeat(num_particles, axis=1),
            debug_y,
            debug_x
        )

        debug_color = jnp.array([1.0, 0.0, 0.0]) # Red
        debug_colors = jnp.broadcast_to(debug_color[None, None, :], (frames, num_particles, 3))

        render_tex = render_tex.at[debug_indices].set(debug_colors)

    # Convert to uint8 and resize, preserving the H:W aspect ratio
    render_tex = jnp.clip(render_tex * 255.0, 0, 255).astype(jnp.uint8)
    out_h = sz * H // max(H, W)
    out_w = sz * W // max(H, W)
    return jax.image.resize(render_tex, (frames, out_h, out_w, 3), method="nearest")


def render_frame(nodes_timeseries, fields_timeseries, frame_nr: int, sz: int = 256) -> np.ndarray:
    """Render a single frame from a stacked timeseries. Returns (H, W, 3) uint8 numpy array.

    Slices the timeseries at frame_nr and calls render_fields. Manually constructs
    a minimal Fields slice to avoid tree_map issues with None f_grid/energy leaves.
    """
    from microcosmos.structs.fields import Fields
    n = nodes_timeseries.position.shape[0]
    i = min(frame_nr, n - 1)
    nodes_f = jax.tree.map(lambda x: x[i:i + 1], nodes_timeseries)
    fields_f = Fields(
        grid_shape=fields_timeseries.grid_shape,
        steric=fields_timeseries.steric[i:i + 1],
        fluid_velocity=fields_timeseries.fluid_velocity[i:i + 1],
        f_grid=None,
        energy=None,
    )
    return render_fields(fields_f, nodes_f, sz=sz)[0]


def render_fields(fields_timeseries, nodes_timeseries, sz=512, animate_energy=False, animate_arrows=True):
    """Render fluid velocity field as RGB frames with velocity magnitude and direction."""
    frames = fields_timeseries.fluid_velocity.shape[0]
    h, w = fields_timeseries.grid_shape

    # Convert to numpy
    fluid_vel = np.array(fields_timeseries.fluid_velocity)  # Shape: (frames, 2, h, w)
    energy = np.array(fields_timeseries.energy)  # Shape: (frames, h, w)
    node_positions = np.array(nodes_timeseries.position)  # Shape: (frames, num_nodes, 2)

    # Compute velocity magnitude
    vel_mag_all = np.sqrt(fluid_vel[:, 0]**2 + fluid_vel[:, 1]**2)
    vel_mag_max = np.max(vel_mag_all)
    if vel_mag_max < 1e-6:
        vel_mag_max = 1.0

    rendered_frames = []

    # Precompute coordinates for quiver
    skip = max(h // 16, 1)
    y_indices = np.arange(skip//2, h, skip)
    x_indices = np.arange(skip//2, w, skip)

    scale_x = sz / w
    scale_y = sz / h

    # Vector visual scaling
    vec_scale = (sz / 32) / vel_mag_max

    for frame_idx in range(frames):
        # Heatmap
        mag = vel_mag_all[frame_idx]
        mag_norm = mag / vel_mag_max
        mag_int = (mag_norm * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(mag_int, cv2.COLORMAP_HOT)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Turn red to blue

        if animate_energy:
            # Add green channel for energy field visualization
            energy_frame = energy[frame_idx]
            energy_norm = energy_frame / (np.max(energy_frame) + 1e-6)
            energy_int = (energy_norm * 255).astype(np.int16)

            heatmap = heatmap.astype(np.int16)  # Allow adding without overflow
            heatmap[..., 1] = np.clip(heatmap[..., 1] + energy_int, 0, 255)
            heatmap = heatmap.astype(np.uint8)

        img = cv2.resize(heatmap, (sz, sz), interpolation=cv2.INTER_NEAREST)

        if animate_arrows:
            # Quiver
            vx = fluid_vel[frame_idx, 0]
            vy = fluid_vel[frame_idx, 1]

            for yi in y_indices:
                for xi in x_indices:
                    v_x = vx[yi, xi]
                    v_y = vy[yi, xi]
                    if v_x**2 + v_y**2 < 1e-10: continue  # Skip near-zero vectors

                    start_x = int((xi + 0.5) * scale_x)
                    start_y = int((yi + 0.5) * scale_y)
                    end_x = int(start_x + v_x * vec_scale)
                    end_y = int(start_y + v_y * vec_scale)

                    cv2.arrowedLine(img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.2)

        # Nodes
        nodes = node_positions[frame_idx]
        for node in nodes:
            nx, ny = node
            px = int(nx * scale_x)
            py = int(ny * scale_y)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)
            cv2.circle(img, (px, py), 2, (0, 255, 255), 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rendered_frames.append(img_rgb)

    return np.array(rendered_frames, dtype=np.uint8)


def animate(
        nodes_timeseries,
        fields_timeseries,
        filename: str,
        subsample: int,
        uniform_color: bool = False,
        verbose: bool = False,
        fps: int = 30,
        animate_filament: bool = True,
        animate_fluid_velocity: bool = True,
        animate_energy: bool = False,
        animate_arrows: bool = True,
        filament_world_size: int = 256,
        fluid_world_size: int = 512,
    ) -> None:

    # Assuming all frames have the same shape
    save_to_gif = False
    save_to_mp4 = True

    # Normalize filename extension to match output format
    base_filename = filename.rsplit('.', 1)[0]
    if save_to_mp4:
        filename = base_filename + '.mp4'

    # 2. Apply proper subsampling using JAX tree_map.
    # This safely slices every array inside the Nodes/Fields objects by the subsample step.
    step = max(1, subsample)
    nodes_subsampled = jax.tree.map(lambda x: x[::step], nodes_timeseries)
    fields_subsampled = jax.tree.map(lambda x: x[::step], fields_timeseries)

    # convert timeseries to frames
    if verbose:
        with jax_timer("Rendering"):
            # Pass the subsampled data to render
            frames = render(nodes_subsampled, fields_subsampled, filament_world_size, uniform_color=uniform_color)
            jax.block_until_ready(frames)
    else:
        frames = render(nodes_subsampled, fields_subsampled, filament_world_size, uniform_color=uniform_color)
        jax.block_until_ready(frames)

    if animate_filament:
        frames = np.array(frames, dtype=np.uint8)
        frame_h, frame_w = frames.shape[1], frames.shape[2]

    # frames = np.array(frames, dtype=np.uint8)
    # frame_h, frame_w = frames.shape[1], frames.shape[2]

    # if save_to_mp4:
    #     if verbose:
    #         print('exporting to video...')
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(
    #         filename,
    #         fourcc,
    #         fps,
    #         (frame_w, frame_h)
    #     )

    #     # Convert JAX arrays to numpy and iterate through rendered frames
    #     for frame in frames:
    #         # Convert RGB to BGR
    #         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         video_writer.write(frame_bgr)
    #     video_writer.release()

    # if save_to_gif:
    #     # Save GIF
    #     gif_filename = base_filename + '.gif'

    #     # Convert frames to PIL Images and save as GIF
    #     pil_frames = [Image.fromarray(frame) for frame in frames]
    #     pil_frames[0].save(
    #         gif_filename,
    #         save_all=True,
    #         append_images=pil_frames[1:],
    #         duration=1000//fps,
    #         loop=0,
    #         optimize=True,
    #         disposal=2
    #     )

    # Also render and save fluid velocity visualization
    if verbose:
        print('Rendering fluid velocity field...')

    # Use the subsampled data for fluid rendering as well to match
    if animate_fluid_velocity:
        # Also render and save fluid velocity visualization
        if verbose:
            print('Rendering fluid velocity field...')

        fluid_frames = render_fields(fields_subsampled, nodes_subsampled, sz=fluid_world_size, animate_energy=animate_energy, animate_arrows=animate_arrows)
        fluid_h, fluid_w = fluid_frames.shape[1], fluid_frames.shape[2]

        if save_to_mp4:
            fluid_mp4 = base_filename + '_fluid.mp4'
            proc = subprocess.Popen([
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{fluid_w}x{fluid_h}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                fluid_mp4,
            ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for frame in fluid_frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()

        if save_to_gif:
            fluid_gif = base_filename + '_fluid.gif'
            fluid_pil_frames = [Image.fromarray(frame) for frame in fluid_frames]
            fluid_pil_frames[0].save(
                fluid_gif,
                save_all=True,
                append_images=fluid_pil_frames[1:],
                duration=1000//fps,
                loop=0,
                optimize=True,
                disposal=2
            )

        if verbose:
            print(f'Fluid velocity visualization saved to: {base_filename}_fluid')


def check_cv2_window_closed(window_name):
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True
    except:
        return True
    return False


def animate_realtime(
        nodes,
        fields,
        dt: float,
        solver_config: dict,
        step_func: callable,
        extra_info_func: callable,
        extra_key_actions: list = [],
        subtitle: str = "",
        animate_energy: bool = False,
    ) -> None:

    out_sz = 512

    sim_speed = 1
    last_time = time.time()
    fps = 0.0

    cv2.namedWindow('Microcosmos', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Microcosmos', out_sz, out_sz)
    cv2.setWindowTitle('Microcosmos', f"Microcosmos: {subtitle}")

    try:
        cv2.displayStatusBar('Microcosmos', "", delayms=0)
        can_set_status_bar = True
    except:
        can_set_status_bar = False

    while True:
        # Simulation speed control
        steps_per_frame = 4 ** sim_speed if sim_speed > 0 else 1
        wait_per_frame = 4 ** -sim_speed if sim_speed < 0 else 1

        for _ in range(steps_per_frame):
            nodes, fields = step_func(nodes, fields, dt, solver_config)

        # Stack list of Nodes into a single Nodes object with stacked arrays
        # tree_map will apply jnp.stack to each field (position, velocity, etc.)
        nodes_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[nodes])
        fields_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[fields])

        # Render fluid and nodes
        frames = render_fields(fields_stacked, nodes_stacked, sz=out_sz, animate_energy=animate_energy)
        jax.block_until_ready(frames)
        frame = np.array(frames, dtype=np.uint8)[0]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Microcosmos', frame)

        # Calculate FPS
        current_time = time.time()
        dt_frame = current_time - last_time
        last_time = current_time
        curr_fps = 1.0 / (dt_frame + 1e-6)
        smoothing = 0.9
        fps = (fps * smoothing) + ((1.0 - smoothing) * curr_fps) if fps != 0.0 else curr_fps

        # Update status bar (if possible)
        if can_set_status_bar:
            extra_info = extra_info_func(nodes, fields)
            if extra_info is None:
                extra_info = ''
            if isinstance(extra_info, (list, tuple)):
                extra_info = ' | '.join(str(x) for x in extra_info)
            if extra_info != '':
                extra_info = ' | ' + extra_info
            cv2.displayStatusBar('Microcosmos', f"FPS: {fps:.1f}" + extra_info, delayms=0)

        # GUI interactions
        win_key = cv2.waitKey(wait_per_frame) & 0xFF
        if win_key in [27]:  # Esc: quit
            break
        elif win_key in [ord('='), ord('+')]:  # +: speed up simulation
            sim_speed += 1
        elif win_key in [ord('-'), ord('_')]:  # -: slow down simulation
            sim_speed -= 1
        elif win_key in [8]:  # Backspace: reset simulation speed
            sim_speed = 0

        # Additional keys
        for key_list, action in extra_key_actions:
            if win_key in key_list:
                nodes, fields = action(nodes, fields)

        if check_cv2_window_closed('Microcosmos'):
            break

    cv2.destroyAllWindows()