from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from qtpy.QtWidgets import QWidget, QLayout, QGroupBox, QGridLayout


if TYPE_CHECKING:
    import napari


class SAM2Subwidget(QGroupBox):
    # Name used for easier access in the main widget
    _name: str = None

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        title: str,
        parent: QWidget,
        tooltip: Optional[str] = None,
        sub_layout: Optional[QLayout] = None,
    ):
        super().__init__(title)
        self.viewer = viewer
        self.parent = parent
        self.tooltip = tooltip
        # Create the layout to be used in this subwidget
        if sub_layout is None:
            self.layout = QGridLayout()
        else:
            self.layout = sub_layout

    @abstractmethod
    def create_widgets(self):
        raise NotImplementedError(
            "create_widgets must be implemented in subclass"
        )

    def initialize_ui(self):
        # Method for subclasses to actually create the interface
        self.create_widgets()
        # Create overall tooltip for this box
        self.setToolTip(self.tooltip)
        # Add the layout to the group box
        self.setLayout(self.layout)
        # Add the group box to the main layout
        self.parent.layout().addWidget(self)
