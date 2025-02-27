from importlib.resources import files
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Optional

import hydra
from hydra import initialize_config_dir
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
import numpy as np
from PIL import Image
from platformdirs import user_cache_dir
from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QComboBox,
    QLabel,
    QLayout,
    QFileDialog,
    QCheckBox,
)
from qtpy import QtCore
import requests
from sam2.build_sam import build_sam2_video_predictor
import torch

from napari_sam2.subwidget import SAM2Subwidget
from napari_sam2.utils import format_tooltip, get_device, configure_cuda

if TYPE_CHECKING:
    import napari

# TODO: Create dict of model types/info here to centralise it
SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_CONFIG_DIR = files("napari_sam2.sam2_configs")
MODEL_DICT = {
    "Base Plus": {
        "filename": "sam2.1_hiera_base_plus.pt",
        "config": SAM2_CONFIG_DIR.joinpath("sam2.1_hiera_b+.yaml"),
    },
    "Large": {
        "filename": "sam2.1_hiera_large.pt",
        "config": SAM2_CONFIG_DIR.joinpath("sam2.1_hiera_l.yaml"),
    },
    "Small": {
        "filename": "sam2.1_hiera_small.pt",
        "config": SAM2_CONFIG_DIR.joinpath("sam2.1_hiera_s.yaml"),
    },
    "Tiny": {
        "filename": "sam2.1_hiera_tiny.pt",
        "config": SAM2_CONFIG_DIR.joinpath("sam2.1_hiera_t.yaml"),
    },
}


class ModelWidget(SAM2Subwidget):
    _name = "model"

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        title: str,
        parent: QWidget,
        tooltip: str,
        sub_layout: QLayout = None,
    ):
        super().__init__(viewer, title, parent, tooltip, sub_layout)

        # Get cache directory from platformdirs
        self.default_download_dir = Path(user_cache_dir("napari-sam2"))
        if not self.default_download_dir.exists():
            self.default_download_dir.mkdir(parents=True)
        # Create temp directory for frames which SAM2 needs for initialisation
        self.frame_temp_dir = self.default_download_dir / "frames"
        self.frame_temp_dir.mkdir(parents=True, exist_ok=True)
        # Var for active download directory (allows reset to default)
        self.download_dir = self.default_download_dir
        # Vars for model path and type
        self.model_path = None
        self.model_type = None
        # Flag for model loaded
        self.model_loaded = False
        # NOTE: This is a tuple of (model_type, low_memory_mode) to ensure proper enabling/disabling
        self.loaded_model = None
        # SAM2 model objects
        self.inference_state = None
        self.sam2_model = None

        self.device = get_device()
        configure_cuda(self.device)

        self.initialize_ui()
        # Connect close event to delete temp frame directory
        self.destroyed.connect(self.on_close)

    def create_widgets(self):
        self.model_label = QLabel("Model Version:")
        self.model_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_DICT.keys())
        self.model_combo.setToolTip(
            format_tooltip(
                "Select which model to use. Note that changing the model will clear any pre-calculated embeddings."
            )
        )

        self.download_loc_btn = QPushButton("Change Model Folder")
        self.download_loc_btn.setToolTip(
            format_tooltip("Change the folder where the model is stored")
        )
        self.download_loc_btn.clicked.connect(self.change_download_loc)
        self.download_loc_text = QLabel(str(self.default_download_dir))
        self.download_loc_text.setMaximumWidth(400)
        self.download_loc_text.setWordWrap(True)

        # self.config_btn = QPushButton("Custom Model")
        # # self.create_custom_model_widget()
        # self.config_btn.setToolTip(
        #     format_tooltip(
        #         "Load a custom SAM2 model, either by config file or config and checkpoint file."
        #     )
        # )
        # self.config_btn.clicked.connect(self.show_custom_model)
        # TODO: This will be a widget with two paths, one for a model checkpoint and one for a config file. The user will be responsible for this.
        # TODO: The name should then be reflected in the model dropdown to still allow for easy switching between models

        self.low_memory_cb = QCheckBox("Low-Memory Mode")
        self.low_memory_cb.setToolTip(
            format_tooltip(
                "Enable this option to reduce memory usage at the cost of slower performance. Recommended for longer videos or lower VRAM GPUs."
            )
        )
        self.low_memory_cb.stateChanged.connect(self.check_model_load_btn)
        # TODO: Connect to func to incorporate model load and embedding button enabling/disabling

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        self.load_btn.setToolTip(
            format_tooltip(
                "Load the selected model for use. Note this will download the model if it does not exist."
            )
        )

        # NOTE: We put this here as it relies on load_btn
        self.model_combo.currentIndexChanged.connect(self.model_changed)
        # This will not trigger above event if already at 0
        # So just set model_type manually here
        self.model_type = self.model_combo.currentText()

        # Add widgets to layout
        self.layout.addWidget(self.model_label, 0, 0, 1, 1)
        self.layout.addWidget(self.model_combo, 0, 1, 1, 1)
        self.layout.addWidget(self.download_loc_btn, 1, 0, 1, 1)
        self.layout.addWidget(self.download_loc_text, 1, 1, 1, 1)
        self.layout.addWidget(self.low_memory_cb, 2, 0, 1, 1)
        self.layout.addWidget(self.load_btn, 2, 1, 1, 1)

    def _check_model_exists(self):
        # Get current model type
        model_type = self.model_type
        # Check if model exists
        model_fname = MODEL_DICT[model_type]["filename"]
        model_path = self.download_dir / model_fname
        return model_path.exists()

    def load_model(self):
        if not self._check_model_exists():
            self.download_model()
        else:
            self.model_path = (
                self.download_dir / MODEL_DICT[self.model_type]["filename"]
            )
            if not self.model_path.exists():
                show_error(
                    f"Model file {self.model_path} does not exist. Please download it first."
                )
        # TODO: Dropdown option for image or video model? Outside scope?
        # TODO: Add UI option for VOS optimised model
        # Tell Hydra to stuff it and look elsewhere
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(
                MODEL_DICT[self.model_type]["config"].parent.resolve()
            )
        ):
            self.sam2_model = build_sam2_video_predictor(
                config_file=str(MODEL_DICT[self.model_type]["config"].stem),
                ckpt_path=str(self.model_path),
                device=self.device,
                vos_optimized=False,  # Setting to True should be faster
                # clear_non_cond_mem_around_input=True,  # Reduces memory usage
                # add_all_frames_to_correct_as_cond=True,  # Improves use of later correction clicks
            )
        # Set currently loaded model, then handle button text
        self.loaded_model = (self.model_type, self.low_memory_cb.isChecked())
        self.check_model_load_btn()

    def download_model(self):
        show_info(f"Downloading {self.model_type} model...")
        model_dict = MODEL_DICT[self.model_type]
        # Download the model
        model_url = f"{SAM2_BASE_URL}/{model_dict['filename']}"
        # Open the URL and get the content length
        req = requests.get(model_url, stream=True)
        req.raise_for_status()
        content_length = int(req.headers.get("Content-Length"))
        self.model_path = self.download_dir / model_dict["filename"]
        # Download the file and update the progress bar
        with open(self.model_path, "wb") as f:
            with progress(
                desc=f"Downloading {self.model_path.name}...",
                total=content_length,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in req.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        # Close request
        req.close()

    def calc_embeddings(self):
        if not self.model_loaded:
            show_error("Please load a model first!")
            return False
        # Reset state if present
        if self.inference_state is not None:
            self.reset_model()
        # Clear all previous prompts
        self.parent.subwidgets["prompt"].prompt_dict = {}
        # TODO: This is only necessary if data is not an MP4 file
        # Double-check the above as notebook is poss out of date?
        # Probably best to keep frame generation stuff separate from custom reader to delay computation
        self.create_frames()
        # Set kwargs based on memory mode option
        kwargs = {}
        if self.low_memory_cb.isChecked():
            kwargs["offload_video_to_cpu"] = True
            kwargs["async_loading_frames"] = True
        else:
            kwargs["offload_video_to_cpu"] = False
            kwargs["async_loading_frames"] = False
        show_info("Calculating embeddings...")

        @thread_worker(
            connect={
                "finished": self.parent.subwidgets[
                    "data"
                ].finish_calc_embeddings,
            }
        )
        def _init_state(self, kwargs):
            with torch.inference_mode(), torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                self.inference_state = self.sam2_model.init_state(
                    video_path=str(self.frame_folder), **kwargs
                )

        _init_state(self, kwargs)

    def change_download_loc(self):
        new_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select Model Directory",
            directory=str(self.download_dir),
        )
        if new_dir != "":
            self.download_dir = Path(new_dir)
            self.download_loc_text.setText(new_dir)

    def create_frames(self):
        # Get existing data
        layer = self.parent.subwidgets["data"].current_layer
        # NOTE: We pull from layer directly rather than file to ensure any in-Napari changes are carried over, avoid re-reading file, and avoid having to track file paths
        # Check if frames already exist for this specific data
        expected_num_frames = layer.data.shape[0]
        # Frame folder
        self.frame_folder = self.frame_temp_dir / layer.name.split(".")[0]
        self.frame_folder.mkdir(exist_ok=True)
        num_frames = len(list(self.frame_folder.glob("*.jpg")))
        if num_frames == expected_num_frames:
            return
        else:
            show_info("Extracting frames from video...")
            # Loop over frames and save them
            # TODO: here or elsewhere, have some check on the input data for 3D-ness
            for i, frame in enumerate(layer.data):
                slice_arr = Image.fromarray(frame)
                slice_arr.save(f"{self.frame_folder}/{i:05d}.jpg")

    def check_model_load_btn(self, model_type: Optional[str] = None):
        if model_type is None:
            model_type = self.model_type
        # Check if model is loaded and set button text accordingly
        if self.loaded_model == (
            self.model_type,
            self.low_memory_cb.isChecked(),
        ):
            self.load_btn.setText("Model Loaded!")
            self.load_btn.setEnabled(False)
            self.model_loaded = True
        else:
            self.load_btn.setText("Load Model")
            self.load_btn.setEnabled(True)
            self.model_loaded = False
        # Reset embeddings button
        self.parent.subwidgets["data"].check_embedding_btn()

    def model_changed(self):
        self.model_type = self.model_combo.currentText()
        self.check_model_load_btn()
        # Delete all embeddings that may exist
        # TODO: Add a yes/no dialog pop-up if any existing embeddings are present
        # Now embeddings_set
        # self.parent.subwidgets["data"].embeddings_dict = {}
        # TODO: Clear any existing model, and reset load model button text to reflect this

    def add_point_prompt(self, object_id, prompt_dict, frame_idx):
        # Add new points to model
        # NOTE: SAM2 notebook says "we need to send all the clicks and their labels (i.e. not just the last click) when calling add_new_points_or_box"
        # So we pass the entire prompt_dict for this object_id for this frame, not just the new point
        # Convert our lists into numpy arrays for SAM2
        points = np.array(
            prompt_dict[object_id][frame_idx]["points"], dtype=np.float32
        )
        labels = np.array(
            prompt_dict[object_id][frame_idx]["labels"], dtype=np.int32
        )
        with torch.inference_mode(), torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            _, out_obj_ids, out_mask_logits = (
                self.sam2_model.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
            )
        self.parent.subwidgets["model"].check_memory()
        # Turn mask logits into a mask that Napari can handle
        out_obj_ids = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        return out_obj_ids

    def reset_model(self):
        # TODO: Add option for less blunt approach, removing larger parts of inference_state dict?
        self.sam2_model.reset_state(self.inference_state)

    def remove_object_from_model(self, object_id):
        self.sam2_model.remove_object(
            self.inference_state, object_id, need_output=False
        )

    def remove_object_from_frame(self, object_id, frame_idx):
        # TODO: Investigate whether we could use need_output=True here
        # And then update the segmentation immediately?
        self.sam2_model.clear_all_prompts_in_frame(
            self.inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            need_output=False,
        )

    def on_close(self):
        # Clean up temp directory
        print("Cleaning up temp directory...")
        shutil.rmtree(self.frame_temp_dir)

    def check_memory(self):
        # Check current VRAM usage and purge inference_state if close to limit
        # Despite changes, each frame consumes some memory so for very long videos we need remove some bits
        # NOTE: This only works with CUDA. Apple is RAM-bound, and CPU is...not great.
        # TODO: Enable this for Mac by checking system memory instead, even though mine is always >90%
        if not self.device.type == "cuda":
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        # If we're using more than 90% of VRAM, clear the inference state
        if free_bytes / total_bytes < 0.1:
            self.reset_model()
            show_info("Approaching GPU memory limit. Resetting model.")
        print(f"Memory usage: {free_bytes / total_bytes:.2f}")
