import flet as ft

from .dataclasses import Image

class PreviewImage(ft.Container):
    def __init__(self, key, path, callback):
        super().__init__(
            key=key,
            image_src=path,
            width=50,
            height=50,
            on_click=callback,
            image_fit=ft.ImageFit.COVER,
            border_radius=10,
        )