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
    format: str

    def get_path(self):
        return self.cache_path + self.name + '.' + self.format


@dataclass
class ColorPalette:
    normal: str
    light: str
    dark: str