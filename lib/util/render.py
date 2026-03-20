import os
import json
from pathlib import Path
from subprocess import call, DEVNULL
import numpy as np

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = os.environ.get(
    'BLENDER_INSTALLATION_PATH', str(Path.home() / 'Downloads')
)
BLENDER_PATH = os.path.join(
    BLENDER_INSTALLATION_PATH, 'blender-3.0.1-linux-x64', 'blender'
)


def _install_blender() -> None:
    if os.path.isfile(BLENDER_PATH):
        return
    os.system('sudo apt-get update')
    os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
    os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
    os.system(
        f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz '
        f'-C {BLENDER_INSTALLATION_PATH}'
    )

def render_all_views(file_path: str, output_folder: str, num_views: int = 150, seed: int = 42) -> bool | None:
    _install_blender()
    blender_exe = os.environ.get('BLENDER_HOME')
    if blender_exe:
        blender_exe = os.path.expanduser(blender_exe)
    if not blender_exe or not os.path.isfile(blender_exe):
        blender_exe = BLENDER_PATH
    if not os.path.isfile(blender_exe):
        raise FileNotFoundError(
            "Blender executable not found. Install Blender 3.0.1 (Linux x64), e.g. from "
            f"{BLENDER_LINK}, extract under {BLENDER_INSTALLATION_PATH}, then set "
            "BLENDER_HOME to the `blender` binary path (see README)."
        )
    # Build camera {yaw, pitch, radius, fov}
    rng = np.random.RandomState(seed)
    yaws = []
    pitchs = []
    offset = (rng.rand(), rng.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        blender_exe,
        '-b',
        '-P',
        os.path.join(
            os.getcwd(),
            'third_party/TRELLIS/dataset_toolkits',
            'blender_script',
            'render.py',
        ),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL, stderr=DEVNULL)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return True

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base: int, n: int) -> float:
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim: int, n: int) -> list[float]:
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim: int, n: int, num_samples: int) -> list[float]:
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n: int, num_samples: int, offset: tuple[float, float] = (0, 0)) -> list[float]:
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]