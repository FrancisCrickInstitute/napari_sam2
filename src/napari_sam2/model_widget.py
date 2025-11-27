import shutil
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import requests
import torch
from hydra import initialize_config_dir
from napari.qt.threading import thread_worker
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from PIL import Image
from platformdirs import user_cache_dir
from qtpy import QtCore
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLayout,
    QPushButton,
    QWidget,
)
from sam2.build_sam import build_sam2_video_predictor
from skimage.util import img_as_ubyte

from napari_sam2.subwidget import SAM2Subwidget
from napari_sam2.utils import configure_cuda, format_tooltip, get_device

if TYPE_CHECKING:
    import napari

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
        self.loaded_model = None
        self.memory_mode = None
        # SAM2 model objects
        self.inference_state = None
        self.sam2_model = None

        self.device = get_device()
        configure_cuda(self.device)

        self.initialize_ui()
        # Connect close event to delete temp frame directory
        self.destroyed.connect(self.on_close)

    def create_widgets(self):
        if self.device.type == "cuda":
            device_txt = f"Device: {self.device.type.upper()} (CUDA Enabled!)"
        else:
            device_txt = (
                f"Device: {self.device.type.upper()} (CUDA Not Available!)"
            )
        self.device_label = QLabel(device_txt)
        self.device_label.setAlignment(
            QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter
        )
        self.device_label.setToolTip(
            format_tooltip(
                """
The device being used for model inference. CUDA (NVIDIA GPU) is recommended but not required.
If you have a GPU but it is not being used, please check your PyTorch installation.
                """
            )
        )

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
        self.low_memory_cb.stateChanged.connect(self.memory_mode_cb_changed)

        self.super_low_memory_cb = QCheckBox("Super Low-Memory Mode")
        self.super_low_memory_cb.setToolTip(
            format_tooltip(
                "Enable this option to reduce memory usage even further. Recommended for very long videos or very low VRAM GPUs. This will be extremely slow."
            )
        )
        self.super_low_memory_cb.stateChanged.connect(
            self.memory_mode_cb_changed
        )

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
        self.layout.addWidget(self.device_label, 0, 0, 1, 2)
        self.layout.addWidget(self.model_label, 1, 0, 1, 1)
        self.layout.addWidget(self.model_combo, 1, 1, 1, 1)
        self.layout.addWidget(self.download_loc_btn, 2, 0, 1, 1)
        self.layout.addWidget(self.download_loc_text, 2, 1, 1, 1)
        self.layout.addWidget(self.low_memory_cb, 3, 0, 1, 1)
        self.layout.addWidget(self.super_low_memory_cb, 4, 0, 1, 1)
        self.layout.addWidget(self.load_btn, 3, 1, 2, 1)

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
                # clear_non_cond_mem_around_input=True,  # Reduces memory usage; currently broken in SAM2
                # Improves use of later correction clicks
                hydra_overrides_extra=[
                    "++model.add_all_frames_to_correct_as_cond=True",
                ],
            )
        # Set currently loaded model, then handle button text
        self.loaded_model = self.model_type
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
        # Double-check the above as notebook is poss out of date?
        # Probably best to keep frame generation stuff separate from custom reader to delay computation
        success = self.create_frames()
        if not success:
            return
        # Set kwargs based on memory mode option
        kwargs = {}
        # NOTE: These are defaults, we're just being explicit
        kwargs["offload_state_to_cpu"] = False
        kwargs["offload_video_to_cpu"] = False
        kwargs["async_loading_frames"] = False
        # Now adjust based on memory mode
        if self.low_memory_cb.isChecked():
            kwargs["offload_video_to_cpu"] = True
            kwargs["async_loading_frames"] = True
        elif self.super_low_memory_cb.isChecked():
            kwargs["offload_state_to_cpu"] = True
            kwargs["offload_video_to_cpu"] = True
            kwargs["async_loading_frames"] = True
        show_info("Calculating embeddings...")

        @thread_worker(
            connect={
                "finished": self.parent.subwidgets[
                    "data"
                ].finish_calc_embeddings,
            }
        )
        def _init_state(self, kwargs):
            with (
                torch.inference_mode(),
                torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ),
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

        # First, squeeze the data to remove singleton dimensions
        squeezed_data = np.squeeze(layer.data)
        squeezed_ndim = squeezed_data.ndim

        # Initialize variables for 5D case
        t_idx = 0
        c_idx = 1

        # Now handle based on squeezed dimensions
        if squeezed_ndim == 2:
            # 2D image
            expected_num_frames = 1
        elif squeezed_ndim == 3:
            if layer.rgb:
                # RGB image
                expected_num_frames = 1
            else:
                # Grayscale image
                expected_num_frames = squeezed_data.shape[0]
        elif squeezed_ndim == 4:
            # Otherwise we hope that first channel is slices/frames
            expected_num_frames = squeezed_data.shape[0]
            expected_num_channels = squeezed_data.shape[1]
            if expected_num_channels not in [1, 3]:
                show_error(
                    "Data has more than 3 channels. Please select a single channel or RGB data."
                )
                return False
        elif squeezed_ndim == 5:
            # 5D data - check for metadata axes order or assume TCZYX
            axes_order = None
            if hasattr(layer, 'metadata') and 'axes' in layer.metadata:
                axes_order = layer.metadata['axes']

            # If no metadata, assume TCZYX order (Time, Channel, Z, Y, X)
            if axes_order is None:
                axes_order = 'TCZYX'

            # Convert axes_order to uppercase for consistency
            axes_order = axes_order.upper()

            # Find indices for T and C in the axes order
            # Validate that both T and C are present
            if 'T' not in axes_order or 'C' not in axes_order:
                show_error(
                    f"5D data requires both Time (T) and Channel (C) dimensions. Found axes: {axes_order}"
                )
                return False

            t_idx = axes_order.find('T')
            c_idx = axes_order.find('C')

            # Validate indices are within bounds
            if t_idx >= squeezed_ndim or c_idx >= squeezed_ndim:
                show_error(
                    f"Invalid axes order '{axes_order}' for {squeezed_ndim}D data."
                )
                return False

            expected_num_frames = squeezed_data.shape[t_idx]
            expected_num_channels = squeezed_data.shape[c_idx]

            if expected_num_channels not in [1, 3]:
                show_error(
                    "Data has more than 3 channels. Please select a single channel or RGB data."
                )
                return False
        else:
            show_error(
                f"Unsupported data dimensions: {squeezed_ndim}D (original: {layer.data.ndim}D). Please use 2D-5D data."
            )
            return False
        # Create frame folder for this specific image layer
        self.frame_folder = self.frame_temp_dir / layer.name.split(".")[0]
        self.frame_folder.mkdir(exist_ok=True)
        frames = list(self.frame_folder.glob("*.jpg"))
        num_frames = len(frames)
        # If the folder already has as many frames as expected, skip
        # NOTE: If a file has the same name and size as something else, this will be wrong
        # But that's unlikely...right?
        if num_frames == expected_num_frames:
            return True
        else:
            show_info("Extracting frames from video...")
            # If any frames do exist, delete them to be safe
            # NOTE: Issues can arise if same file previously used has since been truncated
            if num_frames > 0:
                for i in frames:
                    i.unlink()
            # Loop over frames and save them
            if expected_num_frames == 1:
                # 2D or single-frame image
                # Save the image as a single frame
                if squeezed_data.dtype != np.uint8:
                    # Convert to uint8 for saving as JPEG
                    slice_arr = Image.fromarray(img_as_ubyte(squeezed_data))
                else:
                    slice_arr = Image.fromarray(squeezed_data)
                # TODO: Check image mode and whether this can be written as JPEG
                # 16-bit cannot be written as JPEG
                slice_arr.save(f"{self.frame_folder}/{0:05d}.jpg")
            else:
                # Otherwise loop over and save each frame/slice as SAM2 expects
                # For 4D and 5D data, we need to handle channels appropriately
                if squeezed_ndim == 3:
                    # 3D data: iterate over first dimension (time/z)
                    for i, frame in enumerate(squeezed_data):
                        if frame.dtype != np.uint8:
                            # Convert to uint8 for saving as JPEG
                            slice_arr = Image.fromarray(img_as_ubyte(frame))
                        else:
                            slice_arr = Image.fromarray(frame)
                        slice_arr.save(f"{self.frame_folder}/{i:05d}.jpg")
                elif squeezed_ndim == 4:
                    # 4D data: T, C, Y, X - iterate over time, handle channels
                    for i in range(expected_num_frames):
                        # Extract frame at time i, handling channels
                        frame_data = squeezed_data[i]  # Shape: (C, Y, X)
                        # If single channel, squeeze it out
                        if expected_num_channels == 1:
                            frame_data = frame_data[0]  # Shape: (Y, X)
                        else:
                            # RGB: transpose to (Y, X, C) for PIL
                            frame_data = np.transpose(frame_data, (1, 2, 0))  # Shape: (Y, X, C)

                        if frame_data.dtype != np.uint8:
                            slice_arr = Image.fromarray(img_as_ubyte(frame_data))
                        else:
                            slice_arr = Image.fromarray(frame_data)
                        slice_arr.save(f"{self.frame_folder}/{i:05d}.jpg")
                elif squeezed_ndim == 5:
                    # 5D data: use axes_order to extract frames properly
                    for i in range(expected_num_frames):
                        # Extract frame at time index i
                        if t_idx == 0:
                            frame_data = squeezed_data[i]  # Get time slice
                        else:
                            # Handle cases where T is not the first dimension
                            # Build slice tuple to extract the i-th time frame
                            slices = [slice(None)] * squeezed_ndim
                            slices[t_idx] = i
                            frame_data = squeezed_data[tuple(slices)]

                        # At this point, frame_data should be 4D (C, Z, Y, X) or similar
                        # We need to extract (Y, X) or (Y, X, C) for PIL
                        if expected_num_channels == 1:
                            # Take first channel (along c_idx dimension after removing time)
                            # Adjust c_idx if time dimension was before it
                            adjusted_c_idx = c_idx if t_idx > c_idx else c_idx - 1
                            # Select the channel
                            frame_data = np.take(frame_data, 0, axis=adjusted_c_idx)
                            # Now max project along Z (should be first remaining axis)
                            if frame_data.ndim == 3:  # (Z, Y, X)
                                frame_data = np.max(frame_data, axis=0)  # (Y, X)
                        else:
                            # RGB: extract and arrange properly
                            # frame_data is (C, Z, Y, X) or similar
                            # Max project along Z (find Z axis - should be axis 1 if C is at 0)
                            adjusted_c_idx = c_idx if t_idx > c_idx else c_idx - 1
                            # Assuming remaining structure has Z after C
                            if frame_data.ndim == 4:  # (C, Z, Y, X)
                                # Max project along Z (axis 1)
                                frame_data = np.max(frame_data, axis=1)  # (C, Y, X)
                                frame_data = np.transpose(frame_data, (1, 2, 0))  # (Y, X, C)

                        if frame_data.dtype != np.uint8:
                            slice_arr = Image.fromarray(img_as_ubyte(frame_data))
                        else:
                            slice_arr = Image.fromarray(frame_data)
                        slice_arr.save(f"{self.frame_folder}/{i:05d}.jpg")
        return True

    def check_model_load_btn(self, model_type: str | None = None):
        if model_type is None:
            model_type = self.model_type
        # Check if model is loaded and set button text accordingly
        if self.loaded_model == self.model_type:
            self.load_btn.setText("Model Loaded!")
            self.load_btn.setEnabled(False)
            self.model_loaded = True
        else:
            self.load_btn.setText("Load Model")
            self.load_btn.setEnabled(True)
            self.model_loaded = False
        # Reset embeddings button
        self.parent.subwidgets["data"].check_embedding_btn()

    def memory_mode_cb_changed(self, state):
        if state == QtCore.Qt.Checked:
            if self.sender() == self.low_memory_cb:
                self.super_low_memory_cb.setChecked(False)
                self.memory_mode = "low"
            elif self.sender() == self.super_low_memory_cb:
                self.low_memory_cb.setChecked(False)
                self.memory_mode = "super_low"
        else:
            self.memory_mode = None
        # Now check embedding button
        self.parent.subwidgets["data"].check_embedding_btn()

    def model_changed(self):
        self.model_type = self.model_combo.currentText()
        self.check_model_load_btn()

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
        with (
            torch.inference_mode(),
            torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ),
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
