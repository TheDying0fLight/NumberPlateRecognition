from dataclasses import dataclass
from .components import PreviewImage

@dataclass
class Video:
    cache_path: str
    original_path: str
    name: str


@dataclass
class Image:
    cache_path: str
    name: str
    format: str
    saved: bool = False
    closed: bool = False
    preview_ref: PreviewImage = None

    def get_path(self):
        return self.cache_path + self.name + '.' + self.format

    def selected(self, selected: bool):
        if self.preview_ref is None: return
        self.preview_ref.toggle_selected(selected)

@dataclass
class ColorPalette:
    normal: str
    light: str
    dark: str