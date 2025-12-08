import os
import sys
import spaces
import base64
import tempfile
import hashlib
import re
import uuid
import shutil
from omegaconf import OmegaConf
from typing import Optional, Union, Tuple

import gradio as gr

GUIDEFLOW_YELLOW = "#ccad57"
GUIDEFLOW_BLUE = "#2459c2"
GUIDEFLOW_GREEN = "#8edf9f"

os.environ["CUMM_DISABLE_JIT"] = "1"
os.environ["SPCONV_DISABLE_JIT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- START XVFB GLOBALLY ---
# Check if we are in a headless environment and DISPLAY is not set
if os.environ.get("DISPLAY") is None:
    print("[INFO] Starting Xvfb for headless rendering...")
    from pyvirtualdisplay import Display
    
    # Start Xvfb. visible=0 means headless.
    # size=(1920, 1080) matches your previous xvfb-run settings.
    display = Display(visible=0, size=(1920, 1080))
    display.start()
    
    # Ensure DISPLAY env var is set for subprocesses
    if os.environ.get("DISPLAY") is None:
         # PyVirtualDisplay usually sets this, but fallback if needed
         os.environ["DISPLAY"] = f":{display.display}"
    
    print(f"[INFO] Xvfb started on {os.environ['DISPLAY']}")

# --- LOGO SETUP (BASE64) ---
def image_to_base64(image_path):
    """Encodes an image to a base64 string for direct HTML embedding."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

logo_rel_path = os.path.join("demos", "assets", "logo.png")
logo_abs_path = os.path.join(project_root, logo_rel_path)
logo_src = image_to_base64(logo_abs_path)

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def _download_objaverse_ckpt():
    if not os.path.exists(os.path.join(project_root, 'models', 'model_objaverse.ckpt')):
        os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
        os.system(f'wget https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt -O {os.path.join(project_root, 'models', 'model_objaverse.ckpt')}')

_install_blender()
_download_objaverse_ckpt()

# Attempt import, handle failure gracefully for the demo shell
try:
    from demos.pipeline_fn import GuideFlow3dPipeline
except ImportError:
    GuideFlow3dPipeline = None

pipe = None
cfg = None

# Initialize Pipeline
try:
    cfg_path = os.path.join(project_root, 'config', 'default.yaml')
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        if GuideFlow3dPipeline:
            pipe = GuideFlow3dPipeline().from_pretrained(cfg)
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    pass

output_dir = os.path.join(project_root, "all_outputs")
os.makedirs(output_dir, exist_ok=True)

# --- MAPPING HELPERS ---

# Dictionary mapping static thumbnail images to actual GLB files
THUMB_TO_GLB = {
    # Structure Mesh Examples
    "example_data/thumbs/structure/bench_chair.png": "example_data/structure_mesh/bench_chair.glb",
    "example_data/thumbs/structure/cabinet.png": "example_data/structure_mesh/cabinet.glb",
    "example_data/thumbs/structure/chair.png": "example_data/structure_mesh/chair.glb",
    "example_data/thumbs/structure/giraffe.png": "example_data/structure_mesh/giraffe.glb",
    "example_data/thumbs/structure/motorcycle.png": "example_data/structure_mesh/motorcycle.glb",
    "example_data/thumbs/structure/plane.png": "example_data/structure_mesh/plane.glb",
    
    # Reference Appearance Mesh Examples
    "example_data/thumbs/appearance/B01DA8LC0A.jpg": "example_data/appearance_mesh/B01DA8LC0A.glb",
    "example_data/thumbs/appearance/B01DJH73Y6.png": "example_data/appearance_mesh/B01DJH73Y6.glb",
    "example_data/thumbs/appearance/B0728KSP33.jpg": "example_data/appearance_mesh/B0728KSP33.glb",
    "example_data/thumbs/appearance/B07B4YXNR8.jpg": "example_data/appearance_mesh/B07B4YXNR8.glb",
    "example_data/thumbs/appearance/B07QC84LP1.png": "example_data/appearance_mesh/B07QC84LP1.glb",
    "example_data/thumbs/appearance/B07QFRSC8M.png": "example_data/appearance_mesh/B07QFRSC8M_zup.glb",
    "example_data/thumbs/appearance/B082QC7YKR.png": "example_data/appearance_mesh/B082QC7YKR_zup.glb"
}

# Create a lookup based on basename to be robust against Gradio temp paths
THUMB_BASENAME_TO_GLB = {os.path.basename(k): v for k, v in THUMB_TO_GLB.items()}

# Create reverse lookup for strict example detection
GLB_ABS_PATH_TO_NAME = {}
for k, v in THUMB_TO_GLB.items():
    abs_p = os.path.abspath(os.path.join(project_root, v))
    name_no_ext = os.path.splitext(os.path.basename(v))[0]
    GLB_ABS_PATH_TO_NAME[abs_p] = name_no_ext

def load_mesh_from_thumb(thumb_path: str) -> Optional[str]:
    """Callback to return the GLB path associated with a thumbnail."""
    if not thumb_path:
        return None
    basename = os.path.basename(thumb_path)
    return THUMB_BASENAME_TO_GLB.get(basename, None)

def _ensure_glb_path(result: Union[str, bytes, os.PathLike]) -> str:
    """Normalize various return types from fn() to a .glb file path."""
    if isinstance(result, (str, os.PathLike)):
        path = os.fspath(result)
        if not os.path.exists(path):
            raise gr.Error("Returned mesh path does not exist.")
        return path
    if isinstance(result, (bytes, bytearray)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
        tmp.write(result)
        tmp.flush()
        tmp.close()
        return tmp.name

def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    if not path or not os.path.exists(path):
        return "nocontent"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def get_cache_folder(struct_mesh_path: str) -> str:
    """Determines the output folder name based on the structure mesh."""
    struct_abs = os.path.abspath(struct_mesh_path)
    
    # 1. Check if it is a known example
    # We check both absolute path and if the file hash matches a known example (to be safe)
    # But simplest is just path check for now as examples are static files.
    if struct_abs in GLB_ABS_PATH_TO_NAME:
        return GLB_ABS_PATH_TO_NAME[struct_abs]

    # Also check if basename matches an example (for when gradio moves files around but keeps names?)
    # Actually, safely relying on content hash is better for "New" files.
    
    current_hash = file_sha256(struct_mesh_path)
    
    # 2. Scan existing temp_* folders for matching hash
    # We look for a struct_mesh.hash file inside.
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("temp_"):
                folder_path = os.path.join(output_dir, item)
                if os.path.isdir(folder_path):
                    hash_file = os.path.join(folder_path, "struct_mesh.hash")
                    if os.path.exists(hash_file):
                        try:
                            with open(hash_file, "r") as f:
                                stored_hash = f.read().strip()
                            if stored_hash == current_hash:
                                return item
                        except:
                            continue

    # 3. If not found, create new temp_{id}
    new_id = uuid.uuid4().hex[:8]
    return f"temp_{new_id}"

# @spaces.GPU(duration=360)
def on_run(
    guidance_mode_state: str,
    app_struct_mesh: Optional[str],
    app_ref_mesh: Optional[str],
    app_ref_image: Optional[str],
    sim_struct_mesh: Optional[str],
    sim_ref_text: Optional[str],
    sim_ref_image: Optional[str],
    target_up_label: str,
    reference_up_label: str,
    cfg_strength: float,
    num_steps: int,
    learning_rate: float,
) -> Tuple[str, Optional[str]]:
    
    current_mode = guidance_mode_state.lower()

    if current_mode == "appearance":
        target_mesh_path = app_struct_mesh
        reference_mesh_path = app_ref_mesh
        reference_image_path = app_ref_image
        reference_text = None
    else:
        target_mesh_path = sim_struct_mesh
        reference_text = sim_ref_text
        reference_image_path = sim_ref_image
        reference_mesh_path = None

    if not target_mesh_path:
        raise gr.Error(f"Target Structure mesh is required for {current_mode} mode.")
    
    if pipe is None:
        raise gr.Error("Pipeline not initialized. Check logs.")

    # --- Determine Output Directory ---
    folder_name = get_cache_folder(target_mesh_path)
    run_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"[INFO] Using output directory: {run_output_dir}")

    args = {
        "structure_mesh": target_mesh_path,
        "output_dir": run_output_dir,
        "convert_target_yup_to_zup": target_up_label == "Z-up",
        "convert_appearance_yup_to_zup": reference_up_label == "Z-up",
        "appearance_mesh": reference_mesh_path,
        "appearance_image": reference_image_path,
        "appearance_text": (reference_text or "").strip(),
    }

    fn = None
    if current_mode == "appearance":
        if not reference_mesh_path:
            raise gr.Error("Appearance mode requires a reference mesh.")
        fn = pipe.run_appearance
        args.pop("appearance_text", None)
    else: # similarity
        if not reference_text and reference_image_path:
            args["appearance_image"] = reference_image_path
            args.pop("appearance_text", None)
            args["app_type"] = "image"
        elif reference_text and not reference_image_path:
            args["appearance_text"] = reference_text
            args.pop("appearance_image", None)
            args["app_type"] = "text"
        elif reference_text and reference_image_path:
            raise gr.Error("Similarity mode requires a text prompt or reference image, but not both.")
        else:
            raise gr.Error("Similarity mode requires a text prompt or reference image.")
        fn = pipe.run_self_similarity
        args.pop("appearance_mesh", None)
        args.pop("convert_appearance_yup_to_zup", None)

    if cfg:
        updated_cfg = cfg # OmegaConf.load(cfg)
        updated_cfg.cfg_strength = cfg_strength
        updated_cfg.steps = num_steps
        updated_cfg.learning_rate = learning_rate
        pipe.cfg = updated_cfg

    try:
        result_mesh, result_video = fn(**args)
        mesh_path = _ensure_glb_path(result_mesh)
        video_path = _ensure_glb_path(result_video)
        return mesh_path, video_path
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# --- UI Styling & Header ---

font_reg = os.path.join(project_root, "demos", "assets", "fonts", "avenir-next", "AvenirNextCyr-Regular.ttf")
font_bold = os.path.join(project_root, "demos", "assets", "fonts", "avenir-next", "AvenirNextCyr-Bold.ttf")
font_heavy = os.path.join(project_root, "demos", "assets", "fonts", "avenir-next", "AvenirNextCyr-Heavy.ttf")

css = f"""
@font-face {{
    font-family: 'Avenir Next Regular';
    src: url('/file={font_reg}') format('truetype');
    font-weight: normal;
    font-style: normal;
}}
    @font-face {{
        font-family: 'Avenir Next Bold';
        src: url('/file={font_bold}') format('truetype');
        font-weight: bold;
        font-style: normal;
    }}

@font-face {{
    font-family: 'Avenir Next Heavy';
    src: url('/file={font_heavy}') format('truetype');
    font-weight: normal;
    font-style: normal;
}}

body, .gradio-container {{
    background-color: #ffffff !important;
    color: #1f2937 !important;
    font-family: 'Avenir Next Regular', sans-serif !important;
}}
.dark body, .dark .gradio-container {{
    background-color: #ffffff !important;
    color: #1f2937 !important;
    font-family: 'Avenir Next Regular', 'Inter', 'Roboto', sans-serif !important;
}}
/* Add specific components */
.gradio-container button, 
.gradio-container input, 
.gradio-container textarea, 
.gradio-container label, 
.gradio-container span, 
.gradio-container p, 
.gradio-container h1, 
.gradio-container h2, 
.gradio-container h3, 
.gradio-container h4, 
.gradio-container h5, 
.gradio-container h6
{{
    font-family: 'Avenir Next Regular', sans-serif !important;
}}
.guideflow-header {{
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    margin-bottom: 1rem;
    transform: translateY(0.5rem);
}}
.logo-row {{
    display: flex; 
    align-items: baseline; 
    gap: 0.2rem;
}}
.logo-img {{
    height: 4rem; 
    width: auto;
    transform: translateY(0.5rem);
}}
.title-uide, .title-flow, .title-3d {{
    font-family: 'Avenir Next Regular', sans-serif !important;
    font-size: 3.5rem;
    font-weight: normal;
    line-height: 1.2;
}}
.title-uide {{
    background: linear-gradient(90deg, {GUIDEFLOW_GREEN}, {GUIDEFLOW_BLUE});
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}}
.title-flow {{
    color: {GUIDEFLOW_BLUE};
}}
.title-3d {{
    color: {GUIDEFLOW_YELLOW};
}}
.subtitle {{
    font-size: 1.5rem; 
    font-family: 'Avenir Next Regular', sans-serif;
    color: {GUIDEFLOW_YELLOW}; 
    margin-top: 0.5rem; 
    text-align: center;
}}
.authors {{
    font-size: 1rem; 
    color: #334155; 
    margin-top: 0.5rem;
}}
.affiliations {{
    font-size: 0.9rem; 
    color: #6b7280; 
    margin-top: 0.2rem;
}}
.venue {{
    font-size: 1.1rem; 
    font-weight: 700; 
    color: #111827; 
    margin-top: 0.5rem;
}}
.links a {{
    color: {GUIDEFLOW_BLUE};
    text-decoration: none;
    margin: 0 0.5rem;
    font-weight: 500;
}}
.links a:hover {{
    text-decoration: underline;
}}
.demo-credit {{
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.5rem;
}}
.instructions-container {{
    max-width: 800px;
    margin: 0 auto 2rem auto;
    text-align: left;
    padding: 0 1rem;
}}
.input-row {{ align-items: flex-start; margin-bottom: 1rem; }}
"""

HEADER_HTML = f"""
<div class="guideflow-header">
  <div class="logo-row">
    <img src="{logo_src}" class="logo-img" alt="GuideFlow3D Logo" />
    <span class="title-uide">uide</span><span class="title-flow">Flow</span><span class="title-3d">3D</span>
  </div>
  <div class="subtitle">Optimization-Guided Rectified Flow For Appearance Transfer</div>
  <div class="authors">
    <a href="https://sayands.github.io/" target="_blank">Sayan Deb Sarkar</a><sup>1</sup> &nbsp;&nbsp; 
    <a href="https://vevenom.github.io/" target="_blank">Sinisa Stekovic</a><sup>2</sup> &nbsp;&nbsp; 
    <a href="https://vincentlepetit.github.io/" target="_blank">Vincent Lepetit</a><sup>2</sup> &nbsp;&nbsp; 
    <a href="https://ir0.github.io/" target="_blank">Iro Armeni</a><sup>1</sup>
  </div>
  <div class="affiliations">
    <sup>1</sup>Stanford University &nbsp;&nbsp; <sup>2</sup>ENPC, IP Paris
  </div>
  <div class="venue">NeurIPS 2025</div>
  <div class="links" style="margin-top:10px;">
    <a href="https://arxiv.org/abs/2510.16136" target="_blank">Paper</a> |
    <a href="https://sayands.github.io/guideflow3d" target="_blank">Project Page</a> |
    <a href="https://github.com/sayands/guideflow3d" target="_blank">GitHub</a>
  </div>
  <div class="demo-credit">
    Demo made by <a href="https://suvadityamuk.com" target="_blank" style="color: inherit; text-decoration: underline;">Suvaditya Mukherjee</a>
  </div>
</div>
"""

INSTRUCTIONS_MD = """
<div class="instructions-container">
<h3>Instructions</h3>
<ol>
  <li><strong>Upload a Structure Mesh (.glb):</strong> This defines the shape of your 3D object. We expect a y-up mesh, but feel free to convert it using the "Advanced Settings" below.</li>
  <li><strong>Choose Guidance Mode:</strong> Select "Self-Similarity" (Text) or "Appearance" (Mesh/Image) using the tabs.</li>
  <li><strong>Provide Reference:</strong> Enter a text prompt or upload a reference image/mesh.</li>
  <li><strong>Run:</strong> Click "Generate 3D Asset" to create the result.</li>
  <li><strong>Result:</strong> The result will be displayed in the viewer on the left, and a video will be generated on the right.</li>
</ol>
</div>
"""

# Example Data
EX_STRUCT_THUMBS = [
    ["example_data/thumbs/structure/bench_chair.png"],
    ["example_data/thumbs/structure/cabinet.png"],
    ["example_data/thumbs/structure/chair.png"],
    ["example_data/thumbs/structure/giraffe.png"],
    ["example_data/thumbs/structure/motorcycle.png"],
    ["example_data/thumbs/structure/plane.png"]
]

EX_MESH_THUMBS = [
    ["example_data/thumbs/appearance/B01DA8LC0A.jpg"],
    ["example_data/thumbs/appearance/B01DJH73Y6.png"],
    ["example_data/thumbs/appearance/B0728KSP33.jpg"],
    ["example_data/thumbs/appearance/B07B4YXNR8.jpg"],
    ["example_data/thumbs/appearance/B07QC84LP1.png"],
    ["example_data/thumbs/appearance/B07QFRSC8M.png"],
    ["example_data/thumbs/appearance/B082QC7YKR.png"]
]

EX_IMG = [
    "example_data/appearance_image/B01DA8LC0A.jpg",
    "example_data/appearance_image/B01DJH73Y6.png",
    "example_data/appearance_image/B0728KSP33.jpg",
    "example_data/appearance_image/B07B4YXNR8.jpg",
    "example_data/appearance_image/B07QC84LP1.png",
    "example_data/appearance_image/B07QFRSC8M.jpg",
    "example_data/appearance_image/B082QC7YKR.png"
]
EX_TEXT = ["a wooden chair", "a marble statue", "A black metal-framed bed with a curved headboard, white rectangular mattress, and two white rectangular pillows.", "Rectangular wooden cabinet with reddish-brown finish, standing on four short legs. Features two drawers (upper larger, lower smaller) and an open shelf below. Back has a power socket extension and cable, ideal for electronics."]

with gr.Blocks(
    title="GuideFlow3D",
) as demo:
    
    gr.HTML(HEADER_HTML)
    gr.HTML(INSTRUCTIONS_MD)
    
    guidance_mode_state = gr.State(value="Similarity")

    with gr.Tabs() as guidance_tabs:
        
        # --- TAB 1: SELF-SIMILARITY (LEFT) ---
        with gr.TabItem("Self-Similarity", id="tab_similarity") as tab_sim:
            gr.Markdown("### Similarity Editing Inputs")
            
            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=3):
                    sim_struct_mesh = gr.Model3D(label="Structure Mesh (.glb)", interactive=True, height=300)
                with gr.Column(scale=2):
                    sim_struct_hidden = gr.Image(type="filepath", visible=False)
                    # sim_struct_mesh_examples = gr.Examples(examples=EX_STRUCT_THUMBS, inputs=sim_struct_hidden, label="Structure Examples")
                    sim_struct_mesh_examples = gr.Examples(
                        examples=EX_STRUCT_THUMBS, 
                        inputs=sim_struct_hidden, 
                        outputs=sim_struct_mesh,      # Target the 3D viewer directly
                        fn=load_mesh_from_thumb,      # Run the conversion function
                        run_on_click=True,            # Force execution on click
                        label="Structure Examples"
                    )

            gr.Markdown("> **_NOTE:_**  Please use either a reference image or a reference text prompt, but not both.")

            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=3):
                    sim_ref_image = gr.Image(label="Reference Appearance Image", type="filepath", height=250)
                with gr.Column(scale=2):
                    gr.Examples(examples=EX_IMG, inputs=sim_ref_image, label="Image Examples")
            
            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=3):
                    sim_ref_text = gr.Textbox(label="Reference Text Prompt", placeholder="Describe the appearance...", lines=2)
                with gr.Column(scale=2):
                    gr.Examples(examples=EX_TEXT, inputs=sim_ref_text, label="Prompt Examples")

        # --- TAB 2: APPEARANCE (RIGHT) ---
        with gr.TabItem("Appearance", id="tab_appearance") as tab_app:
            gr.Markdown("### Appearance Transfer Inputs")
            
            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=3):
                    app_struct_mesh = gr.Model3D(label="Structure Mesh (.glb)", interactive=True, height=300)
                with gr.Column(scale=2):
                    app_struct_hidden = gr.Image(type="filepath", visible=False)
                    # app_struct_mesh_examples = gr.Examples(examples=EX_STRUCT_THUMBS, inputs=app_struct_hidden, label="Structure Examples")
                    app_struct_mesh_examples = gr.Examples(
                        examples=EX_STRUCT_THUMBS, 
                        inputs=app_struct_hidden, 
                        outputs=app_struct_mesh,      # Target the 3D viewer directly
                        fn=load_mesh_from_thumb,      # Run the conversion function
                        run_on_click=True,            # Force execution on click
                        label="Structure Examples"
                    )

            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=3):
                    app_ref_image = gr.Image(label="Reference Appearance Image", type="filepath", height=250)
                with gr.Column(scale=2):
                    gr.Examples(examples=EX_IMG, inputs=app_ref_image, label="Image Examples")

            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=3):
                    app_ref_mesh = gr.Model3D(label="Reference Appearance Mesh (.glb)", interactive=True, height=300)
                with gr.Column(scale=2):
                    app_ref_mesh_hidden = gr.Image(type="filepath", visible=False)
                    # app_ref_mesh_examples = gr.Examples(examples=EX_MESH_THUMBS, inputs=app_ref_mesh_hidden, label="Mesh Examples")
                    app_ref_mesh_examples = gr.Examples(
                        examples=EX_MESH_THUMBS, 
                        inputs=app_ref_mesh_hidden, 
                        outputs=app_ref_mesh,         # Target the 3D viewer directly
                        fn=load_mesh_from_thumb,      # Run the conversion function
                        run_on_click=True,            # Force execution on click
                        label="Mesh Examples"
                    )

    # --- ADVANCED SETTINGS ---
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            target_up = gr.Radio(["Y-up", "Z-up"], value="Y-up", label="Structure Mesh Up-Axis")
            reference_up = gr.Radio(["Y-up", "Z-up"], value="Y-up", label="Appearance Mesh Up-Axis")
        
        with gr.Row():
            cfg_strength = gr.Slider(0.1, 10.0, value=5.0, step=0.1, label="CFG Strength")
            num_steps = gr.Slider(50, 1000, value=300, step=50, label="Diffusion Steps")
            learning_rate = gr.Number(value=5e-4, label="Learning Rate")

    # --- RUN BUTTON ---
    with gr.Row():
        run_btn = gr.Button("Generate 3D Asset", variant="primary", size="lg")

    # --- OUTPUTS ---
    gr.Markdown("### Results")
    with gr.Row():
        with gr.Column():
            output_model = gr.Model3D(label="Output Mesh", interactive=False, clear_color=[1.0, 1.0, 1.0, 0.0])
        with gr.Column():
            output_video = gr.Video(label="Output Video", autoplay=True, loop=True, interactive=False)

    tab_sim.select(lambda: "Similarity", outputs=guidance_mode_state)
    tab_app.select(lambda: "Appearance", outputs=guidance_mode_state)

    run_btn.click(
        fn=on_run,
        inputs=[
            guidance_mode_state,
            app_struct_mesh, app_ref_mesh, app_ref_image,
            sim_struct_mesh, sim_ref_text, sim_ref_image,
            target_up, reference_up, cfg_strength, num_steps, learning_rate
        ],
        outputs=[output_model, output_video]
    )
    
    demo.load(None, None, None, js="() => { document.body.classList.remove('dark'); }")

if __name__ == "__main__":
    # demo.queue().launch(share=True, allowed_paths=[project_root], mcp_server=True) # Useful for Colab runs
    demo.queue().launch(
        allowed_paths=[project_root], 
        mcp_server=True, 
        css=css,
        theme=gr.themes.Default(
            primary_hue="sky", 
            secondary_hue="lime"
        ).set(
            body_background_fill="white",
            background_fill_primary="white",
            block_background_fill="white",
            input_background_fill="#f9fafb"
        )
    )
