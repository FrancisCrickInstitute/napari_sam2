from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from napari.utils.notifications import show_error, show_info
from napari.layers import Image
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
)
import pandas as pd

from napari_sam2.subwidget import SAM2Subwidget
from napari_sam2.utils import format_tooltip

if TYPE_CHECKING:
    import napari


class ExportWidget(SAM2Subwidget):
    _name = "export"

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        title: str,
        parent: QWidget,
        tooltip: str,
        sub_layout: QLayout = None,
    ):
        super().__init__(viewer, title, parent, tooltip, sub_layout)

        self.export_dir = Path.home()

        self.initialize_ui()

    def create_widgets(self):
        self.masks_btn = QPushButton("Export Masks")
        self.masks_btn.clicked.connect(self.export_masks)
        self.masks_btn.setToolTip(
            format_tooltip(
                "Export the masks as individual files or as a single mask overlay."
            )
        )

        # Separate layout for the overlay options
        self.overlay_btn = QPushButton("Export Overlay")
        # NOTE: This makes the button span the two rows
        # self.overlay_btn.setSizePolicy(
        #     QSizePolicy.Preferred, QSizePolicy.Preferred
        # )
        self.overlay_btn.clicked.connect(self.export_overlay)
        self.overlay_btn.setToolTip(
            format_tooltip(
                "Export the overlay as a video with the original image and the mask overlay."
            )
        )
        self.overlay_layout = QHBoxLayout()
        self.overlay_form_layout = QFormLayout()
        self.opacity_label = QLabel("Mask Opacity:")
        self.opacity_input = QLineEdit()
        self.opacity_input.setText("0.5")
        self.overlay_form_layout.addRow(self.opacity_label, self.opacity_input)
        self.fps_label = QLabel("FPS:")
        self.fps_input = QLineEdit()
        self.fps_input.setText("30")
        self.overlay_form_layout.addRow(self.fps_label, self.fps_input)
        self.overlay_layout.addWidget(self.overlay_btn)
        self.overlay_layout.addLayout(self.overlay_form_layout)

        # Buttons for Tracks layer creation & export
        self.create_tracks_btn = QPushButton("Create Tracks")
        self.create_tracks_btn.clicked.connect(self.create_tracks)
        self.create_tracks_btn.setToolTip(
            format_tooltip(
                "Create a Tracks layer from the current annotations."
            )
        )
        self.export_tracks_btn = QPushButton("Export Tracks")
        self.export_tracks_btn.clicked.connect(self.export_tracks)
        self.export_tracks_btn.setToolTip(
            format_tooltip("Export the tracks as a CSV file.")
        )

        # Add the widgets to the layout
        self.layout.addWidget(self.masks_btn, 0, 0, 1, 2)
        self.layout.addLayout(self.overlay_layout, 1, 0, 2, 2)
        self.layout.addWidget(self.create_tracks_btn, 3, 0, 1, 1)
        self.layout.addWidget(self.export_tracks_btn, 3, 1, 1, 1)

    def export_masks(self):
        label_layer = self.viewer.layers[
            self.parent.subwidgets["prompt"].label_layer_name
        ]
        if label_layer is None:
            show_error("No active labels layer found.")
            return
        # FIXME: Even though it's slightly different, this is still a bit of overlap with the prompt layer storing
        # Open file dialog to get the save location
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            caption="Export Masks to File",
            directory=str(self.export_dir),
            filter="TIFFs (*.tif *.tiff)",
        )
        if out_path == "":
            return
        # Reset prompt dir so next open is in the same location
        label_layer.save(out_path)

    def export_overlay(self):
        # Use code from holey_segment to create the overlay mp4 to export
        opacity = float(self.opacity_input.text())
        fps = int(self.fps_input.text())

        mask = self.parent.subwidgets["prompt"].get_mask()
        if mask is None:
            show_error("No mask found.")
            return
        video_layer = self.parent.subwidgets["data"].current_layer
        video = video_layer.data

        # Convert video into RGB if it's grayscale
        if not video_layer.rgb:
            video = np.stack((video,) * 3, axis=-1)

        # Convert masks from label array into RGB masks using their visualised colours
        # TODO: Abstract to get_mask_layer?
        label_layer = self.viewer.layers[
            self.parent.subwidgets["prompt"].label_layer_name
        ]
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        all_labels = np.unique(mask)[1:]
        for label in all_labels:
            colour = label_layer.colormap.map(label)
            # OpenCV uses BGR, so reverse the order
            mask_rgb[mask == label] = (colour[:3] * 255).astype(np.uint8)[::-1]

        frame_mask = mask_rgb > 0
        overlay = (
            video * (1 - opacity * frame_mask)
            + (frame_mask * opacity) * mask_rgb
        ).astype(np.uint8)

        # Open file dialog to get the save location
        export_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select Export Directory",
            directory=str(self.export_dir),
        )
        if export_dir == "":
            return
        # Reset prompt dir so next open is in the same location
        self.export_dir = Path(export_dir)

        prefix = Path(self.parent.subwidgets["data"].current_layer.name).stem
        out_path = self.export_dir / f"{prefix}_overlay.mp4"
        out = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (video.shape[2], video.shape[1]),
        )
        for frame in overlay:
            out.write(frame)
        out.release()
        show_info(f"Overlay video saved to {out_path}")

    def create_tracks(self):
        # Grab the current labels layer
        label_layer = self.viewer.layers[
            self.parent.subwidgets["prompt"].label_layer_name
        ]
        if label_layer is None:
            show_error("No active labels layer found.")
            return
        masks = label_layer.data
        # If there is no data, skip and show an error
        if masks.sum() == 0:
            show_error("No masks found in the current layer.")
            return
        # Input for tracks is a (N, D+1) coords array
        # So for us it looks like:
        # [[obj_id1, z1, y1, x1], [obj_id1, z2, y2, x2], [obj_id2, z1, y1, x1], ...]
        # TODO: Ensure can handle TZYX, not just ZYX
        tracks = []
        properties = {}
        # For each object on each frame, get the centroid as the coords
        # Then stack the object ID on the front with appropriate z
        for z in range(masks.shape[0]):
            # Loop over all objects in this frame, skipping the background
            frame_objects = np.unique(masks[z])[1:]
            for obj_id in frame_objects:
                # Get the mask for this object
                mask = masks[z] == obj_id
                # Get the centroid of the mask
                M = cv2.moments(mask.astype(np.uint8))
                c_x = int(M["m10"] / M["m00"])
                c_y = int(M["m01"] / M["m00"])
                # Append the object ID and coords to the tracks
                tracks.append([obj_id, z, c_y, c_x])
        tracks = np.asarray(tracks)
        properties = {
            "obj_id": tracks[:, 0],
        }
        # Create a tracks layer with the data, using our global colour cycle to match the labels
        # NOTE: The colours do match, but by default are additive, and so doesn't look the same
        # Switching to translucent makes the colours exact, but hard to see with the label
        track_layer = self.viewer.add_tracks(
            tracks,
            name="Mask Tracks",
            properties=properties,
            colormaps_dict={
                "obj_id": self.parent.subwidgets["prompt"].global_colour_cycle
            },
        )
        track_layer.color_by = "obj_id"

    def export_tracks(self):
        # NOTE: Napari does not save tracks by default
        # https://github.com/napari/napari/issues/3936
        # We are just saving a CSV file with the tracks data
        # TODO: Then have the ability to read this back and distinguish from point prompts

        # Grab the current tracks layer
        track_layer = self.viewer.layers["Mask Tracks"]
        if track_layer is None:
            show_error("No active tracks layer found.")
            return
        # If there are no tracks, skip and show an error
        df = pd.concat(
            [
                track_layer.features,
                pd.DataFrame(
                    track_layer.data[:, 1:],
                    columns=["axis-0", "axis-1", "axis-2"],
                ),
            ],
            axis=1,
        )
        # Open file dialog to get the save location
        export_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select Export Directory",
            directory=str(self.export_dir),
        )
        if export_dir == "":
            return
        # Reset prompt dir so next open is in the same location
        self.export_dir = Path(export_dir)
        # Save the CSV file, using relevant image
        image_name = Path(
            self.parent.subwidgets["data"].current_layer.name
        ).stem
        df.to_csv(
            self.export_dir / f"{image_name}_mask_tracks.csv", index=False
        )
        show_info(
            f"Tracks data saved to {self.export_dir / f'{image_name}_mask_tracks.csv'}"
        )
