from typing import TYPE_CHECKING

from qtpy.QtGui import QDesktopServices
from qtpy.QtCore import QUrl
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QScrollArea
)

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

        # self.layout().setAlignment(qtpy.QtCore.Qt.AlignTop)

        # Wrap current layout in a container
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container.setLayout(self.container_layout)

        # Scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.container)

        # Main layout just contains the scroll area
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll)
        self.setLayout(main_layout)
        
        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        # Button that links to a video tutorial
        tutorial_button = self.create_button(
            label="Video Tutorial",
            url="??",
            tooltip="Open our video tutorial",
        )
        button_layout.addWidget(tutorial_button)
        # Button that links to the documentation
        docs_button = self.create_button(
            label="Written Guide",
            url="https://github.com/FrancisCrickInstitute/napari_sam2/blob/main/USAGE.md",
            tooltip="Open the documentation",
        )
        button_layout.addWidget(docs_button)
        # Button for reporting issues
        issue_button = self.create_button(
            label="Report Issue",
            url="https://github.com/FrancisCrickInstitute/napari_sam2/issues",
            tooltip="Report an issue on GitHub",
        )
        button_layout.addWidget(issue_button)
        # Add the button layout to the main layout
        self.layout().addLayout(button_layout)

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

    def create_button(self, label: str, url: str, tooltip: str) -> QPushButton:
        def open_url(url):
            QDesktopServices.openUrl(QUrl(url))

        button = QPushButton(label)
        button.setToolTip(tooltip)
        button.clicked.connect(lambda: open_url(url))
        return button
