from typing import TYPE_CHECKING

from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout

from napari_sam2.data_widget import DataWidget
from napari_sam2.model_widget import ModelWidget
from napari_sam2.prompt_widget import PromptWidget
from napari_sam2.export_widget import ExportWidget
from napari_sam2.utils import format_tooltip

if TYPE_CHECKING:
    import napari


class SAM2MainWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.setLayout(QVBoxLayout())
        # self.layout().setAlignment(qtpy.QtCore.Qt.AlignTop)

        # TODO: Add tutorial/help buttons at the top

        self.subwidgets = {}
        self.add_subwidget(
            ModelWidget(
                viewer=self.viewer,
                title="Model",
                parent=self,
                tooltip=format_tooltip("Model parameters"),
            )
        )
        self.add_subwidget(
            DataWidget(
                viewer=self.viewer,
                title="Data",
                parent=self,
                tooltip=format_tooltip("Data input"),
            )
        )
        self.add_subwidget(
            PromptWidget(
                viewer=self.viewer,
                title="Prompts",
                parent=self,
                tooltip=format_tooltip("Prompts for SAM2"),
            )
        )
        self.add_subwidget(
            ExportWidget(
                viewer=self.viewer,
                title="Export",
                parent=self,
                tooltip=format_tooltip("Export options"),
            )
        )

    def add_subwidget(self, widget):
        self.subwidgets[widget._name] = widget
