"""
GuideFlow3D - Interactive Viser GUI
Inspired by SpaceControl3D's Viser-based GUI approach.
"""

import os
import sys
import time
import threading
import subprocess
import tempfile
import uuid
from io import BytesIO

import numpy as np
from PIL import Image
import trimesh
import viser

# ── Project root ──────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["SPCONV_ALGO"] = "native"

# ── Server ────────────────────────────────────────────────────────────────────
server = viser.ViserServer(port=8080)
server.scene.set_up_direction([0.0, 1.0, 0.0])
server.scene.set_environment_map("studio", background=False, environment_intensity=0.8)
server.scene.configure_default_lights(enabled=True)
server.gui.configure_theme(dark_mode=True)

@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    client.camera.position = (2.5, 1.5, 2.5)
    client.camera.look_at = (0.0, 0.0, 0.0)


# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "struct_mesh_handle": None,   # viser GlbHandle for structure mesh
    "output_mesh_handle": None,   # viser GlbHandle for output mesh
    "app_mesh_handle": None,      # viser GlbHandle for appearance ref mesh (in scene)
    "struct_glb_bytes": None,     # raw bytes of loaded structure .glb
    "output_dir": None,           # last run output directory
    "showing_output": False,
    "running": False,
    "sim_image_bytes": None,      # bytes for similarity ref image (upload)
    "app_mesh_bytes": None,       # bytes for appearance ref mesh (upload)
    "app_image_bytes": None,      # bytes for appearance ref image (upload)
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalise_mesh_for_display(glb_bytes: bytes) -> bytes:
    """Centre and unit-scale a GLB for display in Viser."""
    mesh = trimesh.load(BytesIO(glb_bytes), file_type="glb", force="mesh")
    if mesh.vertices.size == 0:
        return glb_bytes
    center = mesh.bounds.mean(axis=0)
    scale = 1.0 / (mesh.bounds[1] - mesh.bounds[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    buf = BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()


def _load_struct_mesh(glb_bytes: bytes) -> None:
    """Replace the structure mesh in the 3D scene."""
    if state["struct_mesh_handle"] is not None:
        state["struct_mesh_handle"].remove()
        state["struct_mesh_handle"] = None
    if state["output_mesh_handle"] is not None:
        state["output_mesh_handle"].remove()
        state["output_mesh_handle"] = None
    state["showing_output"] = False
    state["output_dir"] = None

    normed = _normalise_mesh_for_display(glb_bytes)
    state["struct_mesh_handle"] = server.scene.add_glb(
        "/struct_mesh", glb_data=normed, visible=True
    )
    gui["toggle_btn"].disabled = True
    gui["status_md"].content = "**Structure mesh loaded.** Ready to generate."


def _load_app_mesh(glb_bytes: bytes) -> None:
    """Load the appearance reference mesh into the 3D scene (offset from structure)."""
    if state["app_mesh_handle"] is not None:
        state["app_mesh_handle"].remove()
        state["app_mesh_handle"] = None
    normed = _normalise_mesh_for_display(glb_bytes)
    state["app_mesh_handle"] = server.scene.add_glb(
        "/app_mesh", glb_data=normed, visible=True
    )
    # Offset to the right so structure and appearance mesh don't overlap
    state["app_mesh_handle"].position = (2.5, 0.0, 0.0)


def _image_bytes_to_rgb_array(data: bytes) -> np.ndarray:
    """Decode image bytes (PNG/JPEG) to HWC RGB uint8 array for viser."""
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)


def _show_output_mesh(output_dir: str, guidance_mode: str) -> None:
    """Load and display the generated output mesh."""
    suffix = "app" if guidance_mode == "Appearance" else "sim"
    glb_path = os.path.join(output_dir, f"out_{suffix}.glb")
    if not os.path.exists(glb_path):
        gui["status_md"].content = (
            f"**Error:** Output mesh not found at `{glb_path}`."
        )
        return

    with open(glb_path, "rb") as f:
        out_bytes = f.read()

    normed = _normalise_mesh_for_display(out_bytes)
    if state["output_mesh_handle"] is not None:
        state["output_mesh_handle"].remove()
    state["output_mesh_handle"] = server.scene.add_glb(
        "/output_mesh", glb_data=normed, visible=False
    )
    state["output_dir"] = output_dir
    gui["toggle_btn"].disabled = False
    gui["status_md"].content = (
        f"**Done!** Output saved to `{output_dir}`. "
        "Press **Toggle** to switch between structure / output mesh."
    )


def _guidance_mode_cli(label: str) -> str:
    """Map dropdown labels to run.py --guidance_mode values."""
    return "appearance" if label == "Appearance" else "similarity"


def _write_upload_to_tmp(data: bytes, suffix: str) -> str:
    """Write upload bytes to a named temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=tempfile.gettempdir()
    )
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _run_pipeline(
    struct_path: str,
    guidance_mode: str,
    sim_text: str,
    sim_image_path: str | None,
    app_mesh_path: str | None,
    app_image_path: str | None,
    convert_yup: bool,
    output_dir: str,
    tmp_files: list[str] | None = None,
) -> None:
    """Run run.py as a subprocess in a background thread."""
    state["running"] = True
    gui["generate_btn"].disabled = True
    gui["generate_btn"].label = "Generating..."
    gui["generate_btn"].icon = viser.Icon.LOADER
    gui["generate_btn"].color = "orange"
    gui["progress_bar"].visible = True

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(project_root, "run.py"),
        "--guidance_mode", _guidance_mode_cli(guidance_mode),
        "--structure_mesh", struct_path,
        "--output_dir", output_dir,
    ]
    if convert_yup:
        cmd.append("--convert_yup_to_zup")

    if guidance_mode == "Appearance":
        cmd.extend(["--appearance_mesh", app_mesh_path])
        if app_image_path:
            cmd.extend(["--appearance_image", app_image_path])
    else:  # Similarity
        if sim_image_path:
            cmd.extend(["--appearance_image", sim_image_path])
        else:
            cmd.extend(["--appearance_text", sim_text.strip()])

    gui["status_md"].content = "**Running pipeline...** (this may take several minutes)"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root,
            bufsize=1,
        )

        log_lines: list[str] = []
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            # Keep last 12 lines in the status panel
            display = "\n".join(log_lines[-12:])
            gui["log_md"].content = f"```\n{display}\n```"

        proc.wait()

        if proc.returncode != 0:
            gui["status_md"].content = (
                "**Pipeline failed.** Check the log below for details."
            )
        else:
            _show_output_mesh(output_dir, guidance_mode)

    except Exception as exc:
        gui["status_md"].content = f"**Error:** {exc}"

    finally:
        for path in (tmp_files or []):
            try:
                os.unlink(path)
            except OSError:
                pass
        state["running"] = False
        gui["generate_btn"].disabled = False
        gui["generate_btn"].label = "Generate"
        gui["generate_btn"].icon = viser.Icon.PLAYER_PLAY
        gui["generate_btn"].color = "green"
        gui["progress_bar"].visible = False


# ── GUI layout ────────────────────────────────────────────────────────────────
gui: dict = {}

server.gui.set_panel_label("GuideFlow3D")

# Header
server.gui.add_markdown(
    "## GuideFlow3D\n"
    "*Optimization-Guided Rectified Flow For Appearance Transfer*\n\n"
    "[Paper](https://arxiv.org/abs/2510.16136) · "
    "[Project](https://sayands.github.io/guideflow3d) · "
    "[GitHub](https://github.com/sayands/guideflow3d)",
    order=0,
)

# ── Structure mesh ─────────────────────────────────────────────────────────────
with server.gui.add_folder("1 · Structure Mesh", order=1, expand_by_default=True):
    gui["struct_upload"] = server.gui.add_upload_button(
        "Upload Structure Mesh (.glb)",
        color="blue",
        icon=viser.Icon.UPLOAD,
        mime_type=".glb",
    )
    gui["struct_path_text"] = server.gui.add_text(
        "Or enter file path",
        initial_value=os.path.join(project_root, "examples", "example1.glb"),
        hint="Absolute path to a .glb file",
    )
    gui["load_struct_btn"] = server.gui.add_button(
        "Load from path", color="gray", icon=viser.Icon.FOLDER_OPEN
    )

# ── Guidance mode ──────────────────────────────────────────────────────────────
gui["guidance_mode"] = server.gui.add_dropdown(
    "Guidance Mode",
    options=["Self-Similarity", "Appearance"],
    initial_value="Self-Similarity",
    order=2,
)

# ── Self-Similarity options ────────────────────────────────────────────────────
gui["sim_folder"] = server.gui.add_folder(
    "2 · Self-Similarity Options", order=3, expand_by_default=True
)
with gui["sim_folder"]:
    server.gui.add_markdown(
        "*Use either a text prompt **or** a reference image, not both.*"
    )
    gui["sim_text"] = server.gui.add_text(
        "Text Prompt",
        initial_value="a wooden chair",
        hint="Describe the desired appearance",
    )
    gui["sim_image_upload"] = server.gui.add_upload_button(
        "Upload Reference Image (optional)",
        color="gray",
        icon=viser.Icon.PHOTO,
        mime_type="image/*",
    )
    gui["sim_image_label"] = server.gui.add_markdown("*No image uploaded.*")

# ── Appearance options ─────────────────────────────────────────────────────────
gui["app_folder"] = server.gui.add_folder(
    "2 · Appearance Options", order=3, expand_by_default=True, visible=False
)
with gui["app_folder"]:
    gui["app_mesh_upload"] = server.gui.add_upload_button(
        "Upload Appearance Mesh (.glb)",
        color="blue",
        icon=viser.Icon.UPLOAD,
        mime_type=".glb",
    )
    gui["app_mesh_label"] = server.gui.add_markdown("*No mesh uploaded.*")
    gui["app_image_upload"] = server.gui.add_upload_button(
        "Upload Appearance Image (optional)",
        color="gray",
        icon=viser.Icon.PHOTO,
        mime_type="image/*",
    )
    gui["app_image_label"] = server.gui.add_markdown("*No image uploaded.*")
    # Placeholder for showing uploaded appearance image in GUI (hidden until image set)
    gui["app_image_display"] = server.gui.add_image(
        np.zeros((1, 1, 3), dtype=np.uint8),
        label="Appearance reference",
        visible=False,
    )

# ── Advanced ───────────────────────────────────────────────────────────────────
with server.gui.add_folder("Advanced Settings", order=4, expand_by_default=False):
    gui["convert_yup"] = server.gui.add_checkbox(
        "Convert Y-up → Z-up", initial_value=True
    )
    gui["output_dir_text"] = server.gui.add_text(
        "Output Directory",
        initial_value=os.path.join(project_root, "outputs", "gui_run"),
        hint="Directory where results will be saved",
    )

# ── Generate ───────────────────────────────────────────────────────────────────
gui["generate_btn"] = server.gui.add_button(
    "Generate", color="green", icon=viser.Icon.PLAYER_PLAY, order=5
)

# ── Toggle ─────────────────────────────────────────────────────────────────────
gui["toggle_btn"] = server.gui.add_button(
    "Toggle Structure / Output",
    color="gray",
    icon=viser.Icon.ARROWS_EXCHANGE,
    order=6,
    disabled=True,
)

# ── Status & log ──────────────────────────────────────────────────────────────
gui["progress_bar"] = server.gui.add_progress_bar(
    value=100, animated=True, color="green", order=7, visible=False
)
gui["status_md"] = server.gui.add_markdown(
    "*Load a structure mesh to begin.*", order=8
)
gui["log_md"] = server.gui.add_markdown("", order=9)


# ── Event handlers ────────────────────────────────────────────────────────────

# Guidance mode toggle
@gui["guidance_mode"].on_update
def _on_mode_change(_) -> None:
    is_app = gui["guidance_mode"].value == "Appearance"
    gui["sim_folder"].visible = not is_app
    gui["app_folder"].visible = is_app


# Upload structure mesh
@gui["struct_upload"].on_upload
def _on_struct_upload(event: viser.GuiEvent) -> None:
    uploaded = event.target.value
    state["struct_glb_bytes"] = uploaded.content
    gui["struct_path_text"].value = ""
    gui["status_md"].content = f"**Uploaded:** `{uploaded.name}` — loading…"
    _load_struct_mesh(uploaded.content)


# Load from path
@gui["load_struct_btn"].on_click
def _on_load_struct_path(_) -> None:
    path = gui["struct_path_text"].value.strip()
    if not path or not os.path.exists(path):
        gui["status_md"].content = f"**Error:** File not found: `{path}`"
        return
    with open(path, "rb") as f:
        raw = f.read()
    state["struct_glb_bytes"] = raw
    gui["status_md"].content = f"**Loading:** `{path}`…"
    _load_struct_mesh(raw)


# Upload similarity reference image
@gui["sim_image_upload"].on_upload
def _on_sim_image_upload(event: viser.GuiEvent) -> None:
    uploaded = event.target.value
    state["sim_image_bytes"] = uploaded.content
    gui["sim_image_label"].content = f"*Image:* **{uploaded.name}**"


# Upload appearance mesh
@gui["app_mesh_upload"].on_upload
def _on_app_mesh_upload(event: viser.GuiEvent) -> None:
    uploaded = event.target.value
    state["app_mesh_bytes"] = uploaded.content
    gui["app_mesh_label"].content = f"*Mesh:* **{uploaded.name}**"
    _load_app_mesh(uploaded.content)


# Upload appearance reference image
@gui["app_image_upload"].on_upload
def _on_app_image_upload(event: viser.GuiEvent) -> None:
    uploaded = event.target.value
    state["app_image_bytes"] = uploaded.content
    gui["app_image_label"].content = f"*Image:* **{uploaded.name}**"
    gui["app_image_display"].image = _image_bytes_to_rgb_array(uploaded.content)
    gui["app_image_display"].visible = True


# Toggle structure / output mesh
@gui["toggle_btn"].on_click
def _on_toggle(_) -> None:
    if state["struct_mesh_handle"] is None and state["output_mesh_handle"] is None:
        return
    state["showing_output"] = not state["showing_output"]
    if state["struct_mesh_handle"] is not None:
        state["struct_mesh_handle"].visible = not state["showing_output"]
    if state["output_mesh_handle"] is not None:
        state["output_mesh_handle"].visible = state["showing_output"]


# Generate
@gui["generate_btn"].on_click
def _on_generate(_) -> None:
    if state["running"]:
        return

    guidance_mode = gui["guidance_mode"].value

    # Validate structure mesh
    if state["struct_glb_bytes"] is None:
        path = gui["struct_path_text"].value.strip()
        if not path or not os.path.exists(path):
            gui["status_md"].content = (
                "**Error:** Please load a structure mesh first."
            )
            return
        with open(path, "rb") as f:
            state["struct_glb_bytes"] = f.read()

    # Write structure mesh to temp file
    tmp_files: list[str] = []
    struct_path = _write_upload_to_tmp(state["struct_glb_bytes"], ".glb")
    tmp_files.append(struct_path)

    # Validate and prepare mode-specific inputs
    sim_image_path = None
    app_mesh_path = None
    app_image_path = None

    if guidance_mode == "Self-Similarity":
        sim_text = gui["sim_text"].value.strip()
        if state["sim_image_bytes"]:
            sim_image_path = _write_upload_to_tmp(state["sim_image_bytes"], ".png")
            tmp_files.append(sim_image_path)
            sim_text = ""
        elif not sim_text:
            gui["status_md"].content = (
                "**Error:** Provide a text prompt or upload a reference image."
            )
            return
    else:  # Appearance
        if state["app_mesh_bytes"] is None:
            gui["status_md"].content = (
                "**Error:** Please upload an appearance mesh (.glb)."
            )
            return
        app_mesh_path = _write_upload_to_tmp(state["app_mesh_bytes"], ".glb")
        tmp_files.append(app_mesh_path)
        if state["app_image_bytes"]:
            app_image_path = _write_upload_to_tmp(state["app_image_bytes"], ".png")
            tmp_files.append(app_image_path)
        sim_text = ""

    # Output directory
    run_id = uuid.uuid4().hex[:8]
    base_out = gui["output_dir_text"].value.strip() or os.path.join(
        project_root, "outputs", "gui_run"
    )
    output_dir = f"{base_out}_{run_id}"

    threading.Thread(
        target=_run_pipeline,
        args=(
            struct_path,
            guidance_mode,
            sim_text,
            sim_image_path,
            app_mesh_path,
            app_image_path,
            gui["convert_yup"].value,
            output_dir,
            tmp_files,
        ),
        daemon=True,
    ).start()


# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("GuideFlow3D Viser GUI running at http://localhost:8080")
    print("Press Ctrl+C to exit.")
    while True:
        time.sleep(1.0)
