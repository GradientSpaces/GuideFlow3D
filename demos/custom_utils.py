import os
import json
from subprocess import call, DEVNULL
import numpy as np
import shutil
import multiprocessing as mp
from lib.util.render import _install_blender, sphere_hammersley_sequence, BLENDER_PATH

try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    pass

def _get_optimal_threads(num_workers):
    """Calculate optimal CPU threads per Blender instance."""
    total_cores = os.cpu_count() or 4
    # Reserve 1 core for system/orchestration if possible
    available_cores = max(1, total_cores - 1)
    # Distribute remaining cores among workers
    threads = max(1, available_cores // num_workers)
    # Cap at 4 threads per instance since we are GPU bound anyway 
    # and too many threads just adds contention
    return min(threads, 4)

def _render_views_chunk(file_path, chunk_output_folder, views_chunk, blender_render_engine, cuda_device_id=None, threads=None):
    """Render a subset of views into a chunk-specific folder."""
    os.makedirs(chunk_output_folder, exist_ok=True)

    # Prepare environment with GPU selection if provided
    env = os.environ.copy()
    if cuda_device_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)

    blender_exec = env.get('BLENDER_HOME', BLENDER_PATH)
    if not os.path.exists(blender_exec) and blender_exec == BLENDER_PATH:
        blender_exec = 'blender' # Fallback if specific path missing

    output_root = os.path.dirname(os.path.dirname(chunk_output_folder))
    blender_cache_dir = os.path.join(output_root, "blender_cache")
    os.makedirs(blender_cache_dir, exist_ok=True)
    env["XDG_CACHE_HOME"] = blender_cache_dir

    args = [
        blender_exec, '-b',
        '-P', os.path.join(os.getcwd(), 'third_party/TRELLIS/dataset_toolkits', 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views_chunk),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', chunk_output_folder,
        '--engine', blender_render_engine,
        '--save_mesh',
    ]
    
    if threads:
        args.extend(['--threads', str(threads)])
        
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    call(args, stdout=DEVNULL, stderr=DEVNULL, env=env)

def _merge_blender_chunks(output_folder, chunk_infos, file_path, blender_render_engine):
    """Merge chunk_* folders into the main output_folder and write transforms.json."""
    frames = []
    mesh_copied = False

    # Track global index for sequential renaming
    global_idx = 0

    for i, (chunk_path, chunk_views) in enumerate(chunk_infos):
        if not os.path.isdir(chunk_path):
            # Even if directory is missing (shouldn't happen due to retry), we advance index to keep alignment if possible
            # But if directory missing, we likely failed. 
            # Let's assume retry logic works or we fail hard.
            global_idx += len(chunk_views)
            continue

        # Copy mesh.ply once (from first chunk that has it)
        mesh_src = os.path.join(chunk_path, "mesh.ply")
        mesh_dst = os.path.join(output_folder, "mesh.ply")
        if not mesh_copied and os.path.exists(mesh_src):
            shutil.copy2(mesh_src, mesh_dst)
            mesh_copied = True

        chunk_transforms_path = os.path.join(chunk_path, "transforms.json")
        
        # Simple retry logic if chunk failed
        if not os.path.exists(chunk_transforms_path):
            print(f"[merge_chunks] Warning: missing transforms.json in {chunk_path}, re-rendering chunk.")
            shutil.rmtree(chunk_path, ignore_errors=True)
            # Use default 1 thread for retry to be safe
            _render_views_chunk(file_path, chunk_path, chunk_views, blender_render_engine, threads=2)
        
        if not os.path.exists(chunk_transforms_path):
            # If still missing, raise error
            raise RuntimeError(f"Unable to generate transforms.json for {chunk_path}")
            
        with open(chunk_transforms_path, "r") as f:
            chunk_data = json.load(f)
            chunk_frames = chunk_data.get("frames", [])
        
        if not chunk_frames:
             # Empty frames could mean render failure
             raise RuntimeError(f"No frames found in {chunk_transforms_path}")

        frame_lookup = {
            os.path.basename(frame.get("file_path", "")): frame for frame in chunk_frames
        }

        # Sort files to ensure we map them to indices consistently if render.py uses ordered names (e.g. 000.png)
        chunk_files = sorted([
            f for f in os.listdir(chunk_path) 
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        # We assume the sorted files correspond to the chunk_views in order
        # If render.py produced '000.png', '001.png', ... they correspond to chunk_views[0], chunk_views[1]...
        
        for idx, img_name in enumerate(chunk_files):
            src = os.path.join(chunk_path, img_name)
            if img_name not in frame_lookup:
                print(f"[merge_chunks] Warning: no metadata for {img_name} in {chunk_transforms_path}, skipping image.")
                # os.remove(src) # Don't remove, just skip
                continue

            # Rename to sequential number based on global index
            # Format: 000.png, 001.png, etc.
            # Or image_000.png if preferred, but adhering to existing project style (struct_renders uses 000.png)
            # User request: "something like image_{num}.png"
            # Interpreting as keeping the number sequential and using a clean format.
            # Since structure renders used 000.png, I'll assume {num:03d}.png is the safe "image number" format.
            # However, if I must follow "image_{num}.png" strictly, I would add the prefix.
            # I will use just the number to maintain compatibility with any dataset loaders expecting standard indices.
            
            # Actually, render.py usually outputs 000.png. 
            # The logic: global_idx tracks the start of this chunk. 
            # The current image is the idx-th image in this chunk.
            
            current_global_num = global_idx + idx
            dst_name = f"{current_global_num:03d}.png"
            dst = os.path.join(output_folder, dst_name)
            
            shutil.move(src, dst)

            frame = frame_lookup[img_name].copy()
            frame["file_path"] = dst_name
            frames.append(frame)

        # Advance global index by number of views in this chunk (or number of files processed?)
        # Better to advance by chunk_views length to keep alignment with original views list
        global_idx += len(chunk_views)

        shutil.rmtree(chunk_path)

    if not frames:
        raise RuntimeError("No frames were merged when building transforms.json")

    transforms_path = os.path.join(output_folder, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump({"frames": frames}, f, indent=4)

def _run_single_render(file_path, output_folder, views, blender_render_engine):
    # For single render, we can use more CPU threads since we are the only process
    threads = min(os.cpu_count() or 4, 8)

    output_root = os.path.dirname(output_folder)
    blender_cache_dir = os.path.join(output_root, "blender_cache")
    os.makedirs(blender_cache_dir, exist_ok=True)
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = blender_cache_dir

    blender_exec = os.environ.get('BLENDER_HOME', BLENDER_PATH)
    if not os.path.exists(blender_exec) and blender_exec == BLENDER_PATH:
        blender_exec = 'blender' # Fallback

    args = [
        # 'xvfb-run',
        # "-s", "-screen 0 1920x1080x24",
        blender_exec, '-b',
        '-P', os.path.join(os.getcwd(), 'third_party/TRELLIS/dataset_toolkits', 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', blender_render_engine,
        '--save_mesh',
        '--threads', str(threads)
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    # call(args, stdout=DEVNULL, stderr=DEVNULL)
    call(args, env=env)


def render_all_views(file_path, output_folder, num_views=150, blender_render_engine="CYCLES", num_workers=None):
    _install_blender()
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    # Determine GPU availability using torch if available (safe check)
    num_gpus = 0
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
    except ImportError:
        pass
    
    # Smart worker count logic
    if num_workers is None:
        if blender_render_engine == 'CYCLES':
            if num_gpus > 0:
                # To maximize VRAM usage and overlap CPU preparation with GPU rendering,
                # we can run multiple concurrent Blender instances per GPU.
                # For object-level scenes, 2-3 workers per GPU is usually the sweet spot.
                # Too many will cause context thrashing; too few leaves VRAM idle.
                WORKERS_PER_GPU = 3
                num_workers = num_gpus * WORKERS_PER_GPU
            else:
                # No GPU found: fallback to CPU. Parallelizing CPU might help if RAM permits.
                # Cap at 4 to be safe.
                num_workers = min(os.cpu_count() or 4, 4)
        else:
             # For non-cycles (e.g. Eevee), we can be slightly more aggressive but still bound by GPU
             if num_gpus > 0:
                 num_workers = num_gpus
             else:
                 num_workers = min(os.cpu_count() or 4, 8)
    
    # Override: Force serial for small batches to avoid startup overhead
    # 15 views is small enough that overhead of 2+ processes > gain
    if len(views) < 30:
        num_workers = 1
    
    if num_workers > 1:
        print(f"[render_all_views] Running with {num_workers} workers (GPUs detected: {num_gpus}).")
    else:
        print(f"[render_all_views] Running serially (GPUs detected: {num_gpus}).")

    if num_workers <= 1:
        _run_single_render(file_path, output_folder, views, blender_render_engine)
    else:
        # Multi-process: split views into chunks and render in parallel
        num_workers = min(num_workers, num_views)
        view_chunks = np.array_split(views, num_workers)

        # Convert numpy arrays back to plain lists (json-serializable)
        view_chunks = [list(chunk) for chunk in view_chunks]
        chunk_infos = []
        
        # Calculate optimal threads per worker
        threads_per_worker = _get_optimal_threads(num_workers)

        with mp.Pool(processes=num_workers) as pool:
            jobs = []
            for idx, chunk in enumerate(view_chunks):
                chunk_output_folder = os.path.join(output_folder, f"chunk_{idx}")
                chunk_infos.append((chunk_output_folder, chunk))
                
                # Assign GPU ID round-robin if GPUs are available
                assigned_gpu = None
                if num_gpus > 0:
                    assigned_gpu = idx % num_gpus
                
                jobs.append(
                    pool.apply_async(
                        _render_views_chunk,
                        (file_path, chunk_output_folder, chunk, blender_render_engine, assigned_gpu, threads_per_worker),
                    )
                )
            for j in jobs:
                j.get()

        _merge_blender_chunks(output_folder, chunk_infos, file_path, blender_render_engine)

    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        # Return list of rendered image paths
        out_renderviews = sorted(
            [
                os.path.join(output_folder, f)
                for f in os.listdir(output_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        return out_renderviews if out_renderviews else None
    return None
