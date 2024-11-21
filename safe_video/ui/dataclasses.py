from dataclasses import dataclass

@dataclass
class Video:
    cache_path: str
    original_path: str
    name: str


@dataclass
class Image:
    cache_path: str
    original_path: str
    name: str

@dataclass
class ColorPalate:
    normal: str
    light: str
    dark: str