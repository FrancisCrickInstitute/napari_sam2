from pathlib import Path
import os
from typing import TYPE_CHECKING
import warnings

from app_model.types import KeyCode
from napari.layers.base import no_op
from napari.layers.points._points_constants import Mode
from napari.qt.threading import thread_worker
from napari._qt.layer_controls.widgets._labels import QtLabelControl
from napari.utils.colormaps import label_colormap
from napari.utils.notifications import show_error, show_info
from napari.utils.translations import trans
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QProgressBar,
    QLabel,
)
import skimage.io
import torch
from tqdm import tqdm

from napari_sam2.subwidget import SAM2Subwidget
from napari_sam2.utils import format_tooltip


if TYPE_CHECKING:
    import napari

# Ignore FutureWarning from Napari due to qt_viewer access
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Public access to Window.qt_viewer.*",
)


class PromptWidget(SAM2Subwidget):
    _name = "prompt"

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        title: str,
        parent: QWidget,
        tooltip: str,
        sub_layout: QLayout = None,
    ):
        super().__init__(viewer, title, parent, tooltip, sub_layout)

        self.point_layer_name = "Point Prompts"
        # Default layer name for masks
        self.label_layer_name = "Masks"
        # Default initial directory for storing prompt layers
        self.prompt_dir = Path.home()
        # Container for prompts, key'd by object ID
        # Each has a subdict with a key for each frame, and a list of prompts and their types
        # And whether they have been processed by SAM2, enabling mixing/switching between auto-segment on-click and batching prompts
        self.prompts = {}

        self.moved_point_idxs = None
        self.moved_point_locs = None

        self.cancel_prop = False
        self.worker = None

        self.global_colour_cycle = label_colormap(
            num_colors=49, seed=0.5, background_value=0
        )

        self.initialize_ui()

    def create_widgets(self):
        # Add a label and tickbox for auto-segmenting on click
        self.auto_segment_tickbox = QCheckBox("Auto-Segment on Click")
        self.auto_segment_tickbox.setChecked(True)
        self.auto_segment_tickbox.setToolTip(
            format_tooltip(
                "Automatically trigger SAM2 when a point is added to the Point Prompts layer"
            )
        )
        self.manual_segment_btn = QPushButton("Trigger\nSegmentation")
        self.manual_segment_btn.clicked.connect(self.manual_segment)
        self.manual_segment_btn.setToolTip(
            format_tooltip(
                "Trigger SAM2 segmentation manually. This will only trigger on prompts not yet sent to SAM2."
            )
        )

        # Buttons to handle prompt layers
        self.create_prompt_layers_btn = QPushButton("Create Layers")
        self.create_prompt_layers_btn.clicked.connect(
            self.create_prompt_layers
        )
        self.reset_prompt_layers_slice_btn = QPushButton("Clear/Reset Slice")
        self.reset_prompt_layers_slice_btn.clicked.connect(
            self.reset_prompt_layers_slice
        )
        self.reset_prompt_layers_slice_btn.setToolTip(
            format_tooltip(
                "Clear all prompts and masks for the current slice/frame. Warning: this will delete all current prompts and masks for this slice/frame!"
            )
        )
        self.reset_prompt_layers_btn = QPushButton("Clear/Reset Layers")
        self.reset_prompt_layers_btn.clicked.connect(self.reset_prompt_layers)
        self.reset_prompt_layers_btn.setToolTip(
            format_tooltip(
                "Clear all prompt layers and reset model state. Warning: this will delete all current prompts and masks!"
            )
        )
        self.store_prompt_layers_btn = QPushButton("Store Layers")
        self.store_prompt_layers_btn.clicked.connect(self.store_prompt_layers)
        self.load_prompt_layers_btn = QPushButton("Load Layers")
        self.load_prompt_layers_btn.clicked.connect(self.load_prompt_layers)
        # Buttons/boxes for video propagation
        self.video_propagate_btn = QPushButton("Propagate\nPrompts")
        self.video_propagate_btn.clicked.connect(self.video_propagate)
        self.video_propagate_btn.setToolTip(
            format_tooltip(
                "Propagate the given prompts across the entire video. This may take some time."
            )
        )
        # TODO: Add a spinbox here for max frame number to propagate to
        # Will need to add some validation...
        self.max_frame_spinbox = None
        self.propagate_direction_box = QCheckBox("Reverse\nPropagation")
        self.propagate_direction_box.setChecked(False)
        self.propagate_direction_box.setToolTip(
            format_tooltip(
                "Reverse the direction of propagation across the video"
            )
        )
        self.cancel_prop_btn = QPushButton("Cancel\nPropagation")
        self.cancel_prop_btn.clicked.connect(self.cancel_propagation)
        self.cancel_prop_btn.setToolTip(
            format_tooltip("Cancel the current propagation")
        )

        # Progress bars resize and are annoying so nest a layout for them
        pbar_layout = QHBoxLayout()
        # Add a progress bar for video propagation
        self.propagate_pbar = QProgressBar()
        # We use a tqdm progress bar to get completion estimations, but not for display
        self.propagate_tqdm_pbar = None
        # Create the label associated with the progress bar
        self.pbar_label = QLabel("Progress: [--:--]")
        self.pbar_label.setToolTip(
            format_tooltip("Shows [elapsed<remaining] time for current run.")
        )
        pbar_layout.addWidget(self.pbar_label)
        pbar_layout.addWidget(self.propagate_pbar)

        # Add the buttons to the layout
        self.layout.addWidget(self.auto_segment_tickbox, 0, 0, 1, 3)
        self.layout.addWidget(self.manual_segment_btn, 0, 3, 1, 3)
        self.layout.addWidget(self.create_prompt_layers_btn, 1, 1, 1, 4)
        self.layout.addWidget(self.reset_prompt_layers_slice_btn, 2, 0, 1, 3)
        self.layout.addWidget(self.reset_prompt_layers_btn, 2, 3, 1, 3)
        self.layout.addWidget(self.store_prompt_layers_btn, 3, 0, 1, 3)
        self.layout.addWidget(self.load_prompt_layers_btn, 3, 3, 1, 3)
        self.layout.addWidget(self.video_propagate_btn, 4, 0, 1, 2)
        self.layout.addWidget(self.propagate_direction_box, 4, 2, 1, 2)
        self.layout.addWidget(self.cancel_prop_btn, 4, 4, 1, 2)
        self.layout.addLayout(pbar_layout, 5, 0, 1, 6)

    def create_prompt_layers(self):
        # NOTE: We want to add labels first so that points is "above", and are better seen
        # Best to separate for easier use later when loading prompts
        self.create_label_layer()
        self.create_points_layer()

    def create_label_layer(self):
        # Skip if the layer already exists
        if self.label_layer_name in self.viewer.layers:
            return
        # Get current image to initialize size of mask layer
        current_image = self.parent.subwidgets[
            "data"
        ].image_layer_dropdown.currentText()
        if current_image == "":
            image_shape = (10, 10, 3)  # Placeholder, shouldn't be reached
        else:
            image_shape = self.viewer.layers[current_image].data.shape
            if self.viewer.layers[current_image].rgb:
                image_shape = image_shape[:-1]
        # Ensure that the prompt layer label cmap is the same as for the Points layer
        self.viewer.add_labels(
            np.zeros(image_shape, dtype=np.uint8),
            name=self.label_layer_name,
            colormap=self.global_colour_cycle,
        )

    def create_points_layer(self):
        # Colour cycle for positive and negative prompts
        border_color_cycle = ["red", "green"]
        # Properties container for the points
        # NOTE: We limit to 50 objects for now just because. This will cause an error if more than 50 objects are present
        # But that's a lot for SAM2 so...
        prompt_properties = {
            "positive_prompt": [False, True],
            "object_id": list(range(0, 49)),
        }
        # NOTE: We use the dict to ensure label points to colour, allowing users to skip values (for whatever reason)
        points_layer = self.viewer.add_points(
            None,
            name=self.point_layer_name,
            size=10,
            ndim=3,
            property_choices=prompt_properties,
            border_color="positive_prompt",
            border_color_cycle=border_color_cycle,
            border_width=0.15,
            face_color="object_id",
            face_color_cycle=dict(
                zip(range(0, 49), self.global_colour_cycle.colors[:49])
            ),
        )
        # NOTE: We disable adding points as we want to handle this manually
        points_layer._drag_modes[Mode.ADD] = no_op
        # Add callbacks/bindings to the prompt layers
        points_layer.mouse_drag_callbacks.append(self.on_mouse_click)
        points_layer.mouse_drag_callbacks.append(self.click_drag)
        points_layer.events.data.connect(self.on_data_change)
        # NOTE: These keycodes are what a Labels layer uses to change the label
        # https://github.com/napari/napari/blob/b2f722fd16e08d84cdf5f9af9cbd08b81639bf72/napari/utils/shortcuts.py#L44-L45
        points_layer.bind_key(KeyCode.Equal, self.on_increase_label)
        points_layer.bind_key(KeyCode.Minus, self.on_decrease_label)
        points_layer.bind_key(KeyCode.KeyM, self.select_max_label)
        self._duplicate_colourbox_control(points_layer)

    def _duplicate_colourbox_control(self, layer):
        """
        This is an extremely hacky function to copy over the Labels layer colourbox control to the Points layer.

        Note that this removes the widgets from the Labels layer...

        So really I need to create new blank versions of these widgets
        and then connect the Labels ones to events that updates the Points ones
        """
        labels_layer = self.viewer.layers[self.label_layer_name]
        colourbox_widget = QtLabelControl(
            self.viewer.window.qt_viewer.controls.widgets[labels_layer],
            labels_layer,
        )
        colourbox_layout = QHBoxLayout()
        colourbox_layout.addWidget(colourbox_widget.colorbox)
        colourbox_layout.addWidget(colourbox_widget.selection_spinbox)

        # Now add it to the new points layer
        self.viewer.window.qt_viewer.controls.widgets[layer].layout().addRow(
            trans._("label:"), colourbox_layout
        )

    def remove_objects_from_frames(
        self, object_ids: int | list[int], frame_idxs: int | list[int]
    ):
        # Just make it easier to handle single or multiple object IDs and frame indices
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        if not isinstance(frame_idxs, list):
            frame_idxs = [frame_idxs]

        for obj_id in object_ids:
            # If this object doesn't exist in prompts at all, ensure all traces of it are removed from the mask & model
            if obj_id not in self.prompts:
                # TODO: Refactor alongside on_data_change to avoid code duplication
                label_layer = self.viewer.layers[self.label_layer_name]
                full_arr = label_layer.data
                full_arr[full_arr == obj_id] = 0
                label_layer.data = full_arr
                # Now trigger a complete SAM2 object removal
                self.parent.subwidgets["model"].remove_object_from_model(
                    object_id=obj_id
                )
            # Otherwise we check the relevant frames
            for frame_idx in frame_idxs:
                # If this object doesn't exist on this frame, skip
                if (
                    obj_id not in self.prompts
                    or frame_idx not in self.prompts[obj_id]
                ):
                    continue
                # Remove this object from this mask on this frame
                label_layer = self.viewer.layers[self.label_layer_name]
                full_arr = label_layer.data
                full_arr[frame_idx][full_arr[frame_idx] == obj_id] = 0
                label_layer.data = full_arr
                # Trigger SAM2 to remove object from this frame
                self.parent.subwidgets["model"].remove_object_from_frame(
                    object_id=obj_id, frame_idx=frame_idx
                )
                # Now update our prompt dict to remove all prompts for this object on this frame
                self.prompts[obj_id].pop(frame_idx)
        # Clean up prompts dict if needed
        for obj_id in object_ids:
            if obj_id in self.prompts and len(self.prompts[obj_id]) == 0:
                self.prompts.pop(obj_id)

    def on_data_change(self, event):
        # This is only for handling removal of points from our prompt dict
        # We don't need to handle adding points here as we do that in the on_mouse_click callback
        if event.action == "removing":
            # The data indices are single item numpy arrays, so just extract the item
            idxs = [i.item() for i in event.data_indices]
            # Get our prompt points layer
            layer = event.source
            # Get the object IDs for every point being removed
            object_ids = [
                layer.features.iloc[i].object_id.item() for i in idxs
            ]
            # Get the current frame index
            frame_idx = self.viewer.dims.current_step[0]
            # For every point being removed, access relevant part of our prompt dict and remove the point
            for idx, obj_id in zip(idxs, object_ids):
                # Turn into integer to ensure position matches stored prompt
                point_loc = [int(i) for i in layer.data[idx]]
                self.update_prompt_dict(
                    point_loc,
                    prompt_type=layer.features.iloc[
                        idx
                    ].positive_prompt.item(),
                    object_id=obj_id,
                    action="remove",
                )
            # Trigger SAM2 segmentation for each relevant object ID
            for idx, obj_id in zip(idxs, object_ids):
                # If this object doesn't exist, remove all traces of it from the mask
                if obj_id not in self.prompts:
                    label_layer = self.viewer.layers[self.label_layer_name]
                    full_arr = label_layer.data
                    full_arr[full_arr == obj_id] = 0
                    label_layer.data = full_arr
                    # Now trigger a complete SAM2 object removal
                    self.parent.subwidgets["model"].remove_object_from_model(
                        object_id=obj_id
                    )
                # If there is only one point for this object, remove the mask and skip SAM2
                elif frame_idx not in self.prompts[obj_id]:
                    # Remove this object from this mask on this frame
                    label_layer = self.viewer.layers[self.label_layer_name]
                    full_arr = label_layer.data
                    full_arr[frame_idx][full_arr[frame_idx] == obj_id] = 0
                    label_layer.data = full_arr
                    # Trigger SAM2 to remove object from this frame
                    self.parent.subwidgets["model"].remove_object_from_frame(
                        object_id=obj_id, frame_idx=frame_idx
                    )
                # Otherwise the prompts for this object have changed so we need to resegment
                else:
                    # NOTE: We get more intuitive results if we first clear this object and then re-add all points
                    # Otherwise, the model still has memory of the old points and they have a small lingering effect
                    self.parent.subwidgets["model"].remove_object_from_frame(
                        object_id=obj_id, frame_idx=frame_idx
                    )
                    out_obj_ids = self.parent.subwidgets[
                        "model"
                    ].add_point_prompt(
                        prompt_dict=self.prompts,
                        object_id=obj_id,
                        frame_idx=frame_idx,
                    )
                    # Now insert new mask into Labels layer
                    self.update_prompt_masks(out_obj_ids, frame_idx=frame_idx)
            # Check how the event looks between the two actions and what we get from it
            # Then we can redesign the prompt dicts accordingly to what makes life easier
            # No need to use sets as we are working with very low N here
        elif event.action == "changed":
            # Be explicit with layer and avoid event source stuff
            points_layer = self.viewer.layers[self.point_layer_name]
            # Avoid retrigger
            with points_layer.events.data.blocker():
                # Ensure only triggered when points have been moved
                if self.moved_point_idxs is not None:
                    points_layer.data[list(self.moved_point_idxs)] = (
                        self.moved_point_locs.squeeze()
                    )
                    # Now reset to ensure clean between drags
                    self.moved_point_idxs = None
                    self.moved_point_locs = None
            points_layer.refresh()

    def _prep_point_features(self, prompt_type: bool | int, object_id: int):
        point_layer = self.viewer.layers[self.point_layer_name]
        feature_defaults = point_layer.feature_defaults
        feature_defaults["positive_prompt"] = prompt_type
        feature_defaults["object_id"] = object_id
        point_layer.feature_defaults = feature_defaults
        point_layer.refresh_colors(update_color_mapping=False)

    def click_drag(self, layer, event):
        self.moved_point_idxs = layer.selected_data
        self.moved_point_locs = layer.data.copy()
        yield

    def on_mouse_click(self, layer, event):
        if not layer.mode == "add":
            return
        # Left-click is a positive prompt
        if event.button == 1:
            prompt_type = True
        # Right-click is a negative prompt
        elif event.button == 2:
            prompt_type = False
        # Otherwise ignore (e.g. other click or panning action performed)
        else:
            return
        # Check that we have initialised the model before we do anything
        if not self.parent.subwidgets["data"].embeddings_calcd:
            show_error(
                "Model not initialised, please use the 'Calculate Embeddings' button first!"
            )
            return
        # NOTE: Calling int here always rounds the float position down
        # We need to repeat this behaviour when handling point removal
        point_loc = [
            int(event.position[0]),
            int(event.position[1]),
            int(event.position[2]),
        ]
        label_layer = self.viewer.layers[self.label_layer_name]
        # Now add properties for this point
        self._prep_point_features(prompt_type, label_layer.selected_label)
        # Manually add the point as we have disabled adding
        # But skip if we are in the middle of a drag action
        while event.type == "mouse_move":
            yield
        layer.add(layer.world_to_data(event.position))
        # Update the prompt_dict with the new prompt to be used in SAM2
        self.update_prompt_dict(
            point_loc,
            prompt_type,
            object_id=label_layer.selected_label,
            action="add",
        )
        if self.auto_segment_tickbox.isChecked():
            # Trigger SAM2 segmentation
            out_obj_ids = self.parent.subwidgets["model"].add_point_prompt(
                prompt_dict=self.prompts,
                object_id=label_layer.selected_label,
                frame_idx=point_loc[0],
            )
            # Now insert new mask into Labels layer
            self.update_prompt_masks(out_obj_ids, frame_idx=point_loc[0])
            # Now update the prompt_dict to show that this prompt has been processed
            self.prompts[label_layer.selected_label][point_loc[0]][
                "processed"
            ] = True

    def update_prompt_dict(
        self, point_loc, prompt_type, object_id, action: str = "add"
    ):
        # Extract the XY coords from point_loc array
        point_prompt = [point_loc[2], point_loc[1]]
        # prompt_type is a bool (1 = positive, 0 = negative)
        positive_prompt = int(prompt_type)
        frame_idx = point_loc[0]
        # Insert or retrieve all prompts for this object for this frame (as SAM2 handles this in 2D)
        if object_id in self.prompts:
            # Check if we have points for this frame
            if frame_idx in self.prompts[object_id]:
                if action == "add":
                    # Add point and label to existing arrays
                    self.prompts[object_id][frame_idx]["points"].append(
                        point_prompt
                    )
                    self.prompts[object_id][frame_idx]["labels"].append(
                        positive_prompt
                    )
                    self.prompts[object_id][frame_idx]["processed"] = False
                else:
                    # Remove point and label from existing arrays
                    idx = self.prompts[object_id][frame_idx]["points"].index(
                        point_prompt
                    )
                    self.prompts[object_id][frame_idx]["points"].pop(idx)
                    self.prompts[object_id][frame_idx]["labels"].pop(idx)
                    # If there are no more points for this frame, remove the frame
                    if len(self.prompts[object_id][frame_idx]["points"]) == 0:
                        self.prompts[object_id].pop(frame_idx)
                    # If there are no more frames for this object, remove the object
                    if len(self.prompts[object_id]) == 0:
                        self.prompts.pop(object_id)
            # This won't be reached if action is remove
            else:
                if not action == "add":
                    raise ValueError("How did we get here?")
                self.prompts[object_id][frame_idx] = {
                    "points": [point_prompt],
                    "labels": [positive_prompt],
                    "processed": False,
                }
        else:
            # If object not present and action is remove, do nothing
            if action == "remove":
                return
            # Create new dict for this object_id
            self.prompts[object_id] = {
                frame_idx: {
                    "points": [point_prompt],
                    "labels": [positive_prompt],
                    "processed": False,
                }
            }

    def update_prompt_masks(self, out_obj_ids: dict, frame_idx: int):
        # Update the prompt masks
        label_layer = self.viewer.layers[self.label_layer_name]
        full_arr = label_layer.data
        # Update the mask layer with the new mask
        if full_arr.ndim == 2:
            shape = (full_arr.shape[0], full_arr.shape[1])
        elif full_arr.ndim == 3:
            shape = (full_arr.shape[1], full_arr.shape[2])
        mask_arr = np.zeros(
            shape,
            dtype=np.uint32,
        )
        # Insert label id into mask array
        for obj_id, mask in out_obj_ids.items():
            mask_arr[mask[0] == True] = obj_id
        # Update the mask layer with the new mask for this frame (if relevant)
        if full_arr.ndim == 2:
            full_arr = mask_arr
        else:
            full_arr[frame_idx] = mask_arr
        # We have to do this to trigger Napari event to update the data, can't just insert the frame
        label_layer.data = full_arr

    def manual_segment(self):
        # Check that we have initialised the model before we do anything
        if not self.parent.subwidgets["data"].embeddings_calcd:
            show_error(
                "Model not initialised, please use the 'Calculate Embeddings' button first!"
            )
            return
        # Check that we have point prompts to segment
        if len(self.prompts) == 0:
            show_error("No prompts to segment")
            return
        # Need to grab all prompts that have not been segmented yet
        for obj_id, frame_dict in self.prompts.items():
            for frame_idx, prompt_dict in frame_dict.items():
                if not prompt_dict["processed"]:
                    # Trigger SAM2 segmentation for this object ID
                    out_obj_ids = self.parent.subwidgets[
                        "model"
                    ].add_point_prompt(
                        prompt_dict=self.prompts,
                        object_id=obj_id,
                        frame_idx=frame_idx,
                    )
                    # Now insert new mask into Labels layer
                    self.update_prompt_masks(out_obj_ids, frame_idx)
                    # Now update the prompt_dict to show that this prompt has been processed
                    self.prompts[obj_id][frame_idx]["processed"] = True

    def reset_prompt_layers_slice(self):
        # Clear all prompts and masks for the current slice/frame
        # Get current frame index
        frame_idx = self.viewer.dims.current_step[0]
        # Remove all points from the Points layer for this frame
        point_layer = self.viewer.layers[self.point_layer_name]
        if point_layer.data is not None and len(point_layer.data) > 0:
            # Get indices of points to remove
            idxs_to_remove = np.where(point_layer.data[:, 0] == frame_idx)[0]
            if len(idxs_to_remove) > 0:
                with point_layer.events.data.blocker():
                    point_layer.remove(idxs_to_remove)
                point_layer.refresh()
        # Remove all relevant masks from the Labels layer & prompt dict
        self.remove_objects_from_frames(
            object_ids=list(self.prompts.keys()), frame_idxs=frame_idx
        )

    def reset_prompt_layers(self):
        # NOTE: To avoid triggering data change events, we delete the layers and recreate them
        self.viewer.layers.remove(self.point_layer_name)
        self.viewer.layers.remove(self.label_layer_name)
        # Clear the prompts dict
        self.prompts = {}
        # Now recreate the layers
        self.create_prompt_layers()
        # Reset inference state of the model
        self.parent.subwidgets["model"].reset_model()
        # Reset the progress bar just to clear previous run
        self.reset_pbar()

    def store_prompt_layers(self):
        # Store the prompt layers
        # Prompt user for location to store the prompt layers
        prompt_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select Prompt Directory",
            directory=str(self.prompt_dir),
        )
        if prompt_dir == "":
            return
        # Reset prompt dir so next open is in the same location
        self.prompt_dir = Path(prompt_dir)
        # Get current image name as the root for the prompt layers
        current_image_name = Path(
            self.parent.subwidgets["data"].current_layer.name
        ).stem
        # Save the points layer using Napari's default func
        points_layer = self.viewer.layers[self.point_layer_name]
        points_layer.save(
            str(self.prompt_dir / f"{current_image_name}_points.csv")
        )
        # # Save the labels layer using Napari's default func
        # # NOTE: It retains labels so what we need
        # label_layer = self.viewer.layers[self.label_layer_name]
        # label_layer.save(
        #     str(self.prompt_dir / f"{current_image_name}_labels.tiff")
        # )

    def load_prompt_layers(self):
        if not self.parent.subwidgets["data"].embeddings_calcd:
            show_error(
                "Model not initialised, please use the 'Calculate Embeddings' button first!"
            )
            return
        # Get name of current image to load the prompt layers
        current_image_name = Path(
            self.parent.subwidgets["data"].current_layer.name
        ).stem
        # Prompt user for location to load the prompt layers
        files, _ = QFileDialog.getOpenFileNames(
            self,
            caption="Select Prompt Files",
            directory=str(self.prompt_dir),
            filter=f"Prompt Files ({current_image_name}_labels.tiff {current_image_name}_points.csv)",
        )
        # TODO: Expand filter if we are storing prompt dict as a pkl
        # Do we add current image name to filter so only 1 can be chosen?

        # NOTE:
        # We only really want users to load 1 set of prompt layers at a time
        # And they only have to provide the points file at a minimum
        # If no labels, we can use the points to regenerate the labels
        # If labels are provided, we will need to add them all as masks to SAM2
        # Though the add_mask func is not really used so may lead to diff results
        # In order to regain the inference state
        # Note that it might be better if we have a way of storing the complete model state to reload that

        if len(files) == 0:
            return
        elif len(files) > 2:
            show_error("Please only select 1 or 2 files")
            return
        else:
            # Separate the files into points and labels files
            label_file = [Path(i) for i in files if i.endswith("_labels.tiff")]
            point_file = [Path(i) for i in files if i.endswith("_points.csv")]
            # NOTE: We'll need to create prompt layers if not already done
            # If already done, we'll need to ensure they are cleared before loading
            # This should maybe require a user prompt to confirm if already have masks but whatever
            # We'll also need to ensure that the model is initialised before loading prompts
            #
            if len(point_file) == 0:
                show_error("A points file is required")
                return
            elif len(point_file) == 1:
                self.load_point_prompts(point_file[0])
            else:
                raise ValueError("How did we get here?")
            if len(label_file) == 1:
                self.load_label_prompts(label_file[0])

    def load_point_prompts(self, point_file: os.PathLike):
        # Need to load the points layer
        # Easiest way to do this is to add one by one, setting feature defaults in between object id or prompt type changes
        # Though maybe try a bulk load first?
        if self.point_layer_name not in self.viewer.layers:
            self.create_points_layer()
        else:
            # If the layer already exists, remove it first to delete anything existing
            self.viewer.layers.remove(self.point_layer_name)
            self.create_points_layer()
        point_layer = self.viewer.layers[self.point_layer_name]
        # Also need to store in the prompt dict? Or maybe that's easiest to just pickle on storing and then load back in...
        df_points = pd.read_csv(point_file)
        for (obj_id, frame_idx), df_group in df_points.groupby(
            ["object_id", "axis-0"]
        ):
            # Need to convert here as they are singleton numpy float arrays which makes things unhappy
            obj_id = int(obj_id)
            frame_idx = int(frame_idx)
            for _, row in df_group.iterrows():
                point_loc = [
                    int(row["axis-0"]),
                    int(row["axis-1"]),
                    int(row["axis-2"]),
                ]
                # Ensure that the prompt type is a bool
                prompt_type = bool(row["positive_prompt"])
                # Setup points layer for proper insertion
                self._prep_point_features(prompt_type, obj_id)
                # Add the point to the layer
                point_layer.add(point_loc)
                # Then add this point into prompt dict
                self.update_prompt_dict(
                    point_loc,
                    prompt_type,
                    object_id=obj_id,
                    action="add",
                )
            # Now trigger SAM2 segmentation for this object ID
            out_obj_ids = self.parent.subwidgets["model"].add_point_prompt(
                prompt_dict=self.prompts,
                object_id=obj_id,
                frame_idx=frame_idx,
            )
            # Now insert new mask into Labels layer
            self.update_prompt_masks(out_obj_ids, frame_idx)

    def load_label_prompts(self, label_file: os.PathLike):
        # First check if the label layer exists
        if self.label_layer_name not in self.viewer.layers:
            self.create_label_layer()
        else:
            self.viewer.layers.remove(self.label_layer_name)
        # Load the data and insert into the label layer
        mask_arr = skimage.io.imread(label_file)
        self.viewer.layers[self.label_layer_name].data = mask_arr

    def on_increase_label(self, layer):
        # Increase the label for the selected point
        label_layer = self.viewer.layers[self.label_layer_name]
        label_layer.selected_label += 1

    def on_decrease_label(self, layer):
        # Decrease the label for the selected point
        label_layer = self.viewer.layers[self.label_layer_name]
        label_layer.selected_label -= 1

    def select_max_label(self, layer):
        # Get the next available label
        label_layer = self.viewer.layers[self.label_layer_name]
        new_max_label = np.max(label_layer.data) + 1
        # NOTE: Napari has a show_info pop-up in this scenario saying 'no'
        if label_layer.selected_label == new_max_label:
            label_layer.selected_label = 1
        else:
            label_layer.selected_label = new_max_label

    def video_propagate(self):
        # Reset flag to cancel the propagation
        # NOTE: We do it here to ensure after final yield we still reset the progress bar
        self.cancel_prop = False
        # Reset current progress bar
        self.reset_pbar()
        # This should just be a trigger of SAM2 propagation
        video_segments = {}
        # Get the first frame that we have prompts for
        first_frame = min(
            [
                min(self.prompts[obj_id].keys())
                for obj_id in self.prompts.keys()
            ]
        )
        # Now propagate the prompts across the video from this frame
        # In the direction specified by the checkbox
        # TODO: Expose max_frame_num_to_track to potentially limit the number of frames to propagate
        sam2_model = self.parent.subwidgets["model"].sam2_model
        inference_states = self.parent.subwidgets["model"].inference_states
        # Initialise progress bars with the number of frames to propagate
        reverse = self.propagate_direction_box.isChecked()
        if reverse:
            # If we include max_frame_num_to_track, max(first_frame - max_num, 0)
            num_frames = first_frame
        else:
            # FIXME: num_frames consistent across states, so take first (0 or 1?)
            num_frames = inference_states[0]["num_frames"] - first_frame
        self.init_pbar(num_frames=num_frames)

        @thread_worker(
            connect={
                "yielded": self.update_propagation,
                "finished": lambda: show_info("Propagation complete!"),
            }
        )
        def _run_propagation(
            sam2_model,
            inference_states,
            first_frame,
            reverse,
            device,
            low_memory,
        ):
            with (
                torch.inference_mode(),
                torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                # TODO: Refactor when figured out how to manage states
                for object_id, inference_state in inference_states.items():
                    for (
                        out_frame_idx,
                        out_obj_ids,
                        out_mask_logits,
                    ) in sam2_model.propagate_in_video(
                        inference_state,
                        start_frame_idx=first_frame,
                        reverse=reverse,
                    ):
                        video_segments[(object_id, out_frame_idx)] = {
                            out_obj_id: (out_mask_logits[i] > 0.0)
                            .cpu()
                            .numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                        yield video_segments, out_frame_idx, object_id
                        self.parent.subwidgets["model"].check_memory()
                        if low_memory:
                            frame_to_remove = (
                                min(
                                    out_frame_idx + 1,
                                    inference_state["num_frames"],
                                )
                                if reverse
                                else max(out_frame_idx - 1, 0)
                            )
                            sam2_model._clear_non_cond_mem_around_input(
                                inference_state, frame_to_remove
                            )
                            print(
                                "Cleared memory around frame", frame_to_remove
                            )
                            self.parent.subwidgets["model"].check_memory()

        # Switch points layer to PAN_ZOOM mode to discourage segmenting while propagating
        self.viewer.layers[self.point_layer_name].mode = "PAN_ZOOM"

        self.worker = _run_propagation(
            sam2_model,
            inference_states,
            first_frame,
            reverse,
            device=self.parent.subwidgets["model"].device,
            low_memory=self.parent.subwidgets["model"].memory_mode
            in ["low", "super_low"],
        )

    def update_propagation(self, outputs):
        video_segments, out_frame_idx, object_id = outputs
        # Update progress bar
        self.update_pbar()
        # Update mask layer with the new masks
        # FIXME: That probs isn't how video_segments is structured, probs default ID (0)
        self.update_prompt_masks(
            out_obj_ids=video_segments[(object_id, out_frame_idx)],
            frame_idx=out_frame_idx,
        )
        # Move Napari viewer to this frame to show the new masks
        self.viewer.dims.set_point(0, out_frame_idx)
        # If cancel btn has been clicked since last yield, cancel the next propagation
        if self.cancel_prop:
            self.worker.quit()
            self.reset_pbar()
            show_info("Propagation cancelled!")

    def init_pbar(self, num_frames: int):
        self.propagate_pbar.setRange(0, num_frames)
        self.propagate_pbar.setValue(0)
        self.propagate_tqdm_pbar = tqdm(total=num_frames)
        self.pbar_label.setText("Progress: [--:--]")

    def update_pbar(self):
        # NOTE: Reversing and starting in odd places means we update pbar incrementally without frame info
        self.propagate_tqdm_pbar.update(1)
        self.propagate_pbar.setValue(self.propagate_tqdm_pbar.n)
        elapsed = self.propagate_tqdm_pbar.format_dict["elapsed"]
        rate = self.propagate_tqdm_pbar.format_dict["rate"]
        # Rate can be None
        rate = rate if rate else 1
        remaining = max(
            (
                (self.propagate_tqdm_pbar.total - self.propagate_tqdm_pbar.n)
                / rate
            ),
            0,
        )
        self.pbar_label.setText(
            f"Progress: [{self.propagate_tqdm_pbar.format_interval(elapsed)}<{self.propagate_tqdm_pbar.format_interval(remaining)}]"
        )

    def reset_pbar(self):
        # Reset our propagation progress bar
        self.propagate_pbar.setValue(0)
        # Handle first run where no tqdm pbar exists
        if self.propagate_tqdm_pbar is not None:
            self.propagate_tqdm_pbar.close()
        self.pbar_label.setText("Progress: [--:--]")

    def get_mask(self):
        # Get the mask for the current frame
        label_layer = self.viewer.layers[self.label_layer_name]
        if label_layer is None:
            return None
        else:
            return label_layer.data

    def cancel_propagation(self):
        # Set a flag to cancel the propagation
        self.cancel_prop = True
