from dataclasses import dataclass, field
from enum import Flag, auto


@dataclass
class FileVersion:
    name: str
    fmt: str


@dataclass
class FileVersionTemplate:
    name: str
    fmt: str | None = None
    max_size: int | None = None

class Version(Flag):
    ORIG = auto()
    PREVIEW = auto()
    ICON = auto()
    BLUR_ORIG = auto()
    BLUR_PREVIEW = auto()


@dataclass
class Media:
    id: str  # name + number to make it unique
    cache_path: str
    name: str
    files: dict[Version, FileVersion] = field(default_factory=lambda: {v: None for v in Version})
    saved: bool = False
    has_to_be_closed: bool = False
    current_preview: Version = Version.PREVIEW
    preview_container = None

    def set_file(self, version: Version, file: FileVersion):
        self.files[version] = file

    def get_path(self, version: Version):
        return f'{self.cache_path}{self.id}/{self.files[version].name}.{self.files[version].fmt}'

    def get_path_preview(self):
        return self.get_path(self.current_preview)

    def get_orig_name(self):
        return f'{self.name}.{self.files[Version.ORIG].fmt}'

    def selected(self, selected: bool):
        if self.preview_container is None:
            return
        self.preview_container.toggle_selected(selected)


class Image(Media):
    def __init__(self, media: Media):
        super().__init__(*media.__dict__.values())


class Video(Media):
    def __init__(self, media: Media, aspect_ratio):
        self.aspect_ratio: float = aspect_ratio
        self.position: float = 0
        super().__init__(*media.__dict__.values())


@dataclass
class ColorPalette:
    normal: str
    background: str
    dark: str
    selected: str
    text: str
