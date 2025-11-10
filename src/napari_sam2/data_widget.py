from pathlib import Path
from typing import TYPE_CHECKING, Optional

from napari.layers import Image
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QPushButton, QComboBox, QLabel, QLayout

from napari_sam2.subwidget import SAM2Subwidget
from napari_sam2.utils import format_tooltip

if TYPE_CHECKING:
    import napari


class DataWidget(SAM2Subwidget):
    _name = "data"

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        title: str,
        parent: QWidget,
        tooltip: str,
        sub_layout: QLayout = None,
    ):
        super().__init__(viewer, title, parent, tooltip, sub_layout)

        self.viewer.layers.events.inserted.connect(self.layer_added)
        self.viewer.layers.events.removed.connect(self.layer_removed)
        self.viewer.layers.selection.events.changed.connect(
            self.switch_selected_layer
        )
        # Store a tuple of (model_type, layer_name) for the current embeddings to avoid recalculating on UI changes
        self.current_embeddings = None
        # Easy reference for current image/video layer being used
        self.current_layer = None
        # Flag to check if embeddings have been calculated
        self.embeddings_calcd = False

        self.initialize_ui()

    def create_widgets(self):
        self.embeddings_btn = QPushButton("Calculate Embeddings")
        self.embeddings_btn.clicked.connect(self.calc_embeddings)
        self.embeddings_btn.setToolTip(
            format_tooltip(
                "Calculate embeddings for the selected image layer. This may take time for larger images/longer videos."
            )
        )

        self.image_layer_label = QLabel("Image Layer:")
        self.image_layer_dropdown = QComboBox()
        self.image_layer_dropdown.currentIndexChanged.connect(
            self.switch_selected_layer
        )
        self.image_layer_dropdown.setMaximumWidth(300)
        # Add any existing layers
        if len(self.viewer.layers) > 0:
            current_image_layers = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
            ]
            if len(current_image_layers) > 0:
                self.image_layer_dropdown.addItems(current_image_layers)

        # Add the widgets to the layout
        self.layout.addWidget(self.image_layer_label, 0, 0, 1, 1)
        self.layout.addWidget(self.image_layer_dropdown, 0, 1, 1, 1)
        self.layout.addWidget(self.embeddings_btn, 1, 0, 1, 2)

    def calc_embeddings(self):
        # Due to how things are initialized, we wrap model subwidget function here
        self.parent.subwidgets["model"].calc_embeddings()

    def finish_calc_embeddings(self):
        show_info("Embeddings calculated!")
        self.embeddings_calcd = True
        # Disable the button after embeddings have been calculated to avoid unnecessary comp
        self.embeddings_btn.setEnabled(False)
        self.embeddings_btn.setText("Embeddings loaded!")
        self.current_embeddings = (
            self.parent.subwidgets["model"].loaded_model,
            self.current_layer.name,
            self.parent.subwidgets["model"].memory_mode,
        )
        # Set max value for number of frames to propagate (total num frames - 1 because the starting frame isn't included)
        self.parent.subwidgets[
                "prompt"
            ].max_frame_spinbox.setMaximum(
                self.parent.subwidgets["model"].inference_state["num_frames"]-1
            )
        
        self.parent.subwidgets[
                "prompt"
            ].start_frame_spinbox.setMaximum(
                self.parent.subwidgets["model"].inference_state["num_frames"]
            )    

    def layer_added(self, event):
        if isinstance(event.value, Image):
            self.image_layer_dropdown.insertItem(0, event.value.name)

    def layer_removed(self, event):
        if isinstance(event.value, Image):
            self.image_layer_dropdown.removeItem(
                self.image_layer_dropdown.findText(event.value.name)
            )

    def switch_selected_layer(self, event):
        # Integer means we're switching layers via the dropdown, not selection
        if isinstance(event, int):
            dropdown_name = self.image_layer_dropdown.itemText(event)
            if dropdown_name == "":
                self.current_layer = None
                return
            selected_layer = self.viewer.layers[dropdown_name]
        else:
            selected_layer = event.source.active
            if selected_layer is None:
                return
            if isinstance(selected_layer, Image):
                # Make sure the dropdown is in sync with the selected layer
                # But only do this if our event source is on insertion (to avoid recursion)
                self.image_layer_dropdown.setCurrentIndex(
                    self.image_layer_dropdown.findText(selected_layer.name)
                )
        # Only update current layer if it's an Image layer
        if isinstance(selected_layer, Image):
            self.current_layer = selected_layer
        # Otherwise, it's a prompt layer (probably) so we don't need to do anything
        else:
            return
        # Check whether we have embeddings for this layer or not
        self.check_embedding_btn()

    def check_embedding_btn(
        self,
        model_type: Optional[str] = None,
        layer_name: Optional[str] = None,
        memory_mode: Optional[str] = None,
    ):
        # Allow to be None so we can extract directly from subwidgets
        if model_type is None:
            loaded_model = self.parent.subwidgets["model"].loaded_model
        if layer_name is None:
            layer_name = (
                self.current_layer.name
                if self.current_layer is not None
                else None
            )
        if memory_mode is None:
            memory_mode = self.parent.subwidgets["model"].memory_mode
        # NOTE: We do not consider low memory mode here as that does not affect the embeddings
        if self.current_embeddings == (loaded_model, layer_name, memory_mode):
            self.embeddings_btn.setEnabled(False)
            self.embeddings_btn.setText("Embeddings loaded!")
            self.embeddings_calcd = True
        else:
            self.embeddings_btn.setEnabled(True)
            self.embeddings_btn.setText("Calculate Embeddings")
            self.embeddings_calcd = False
