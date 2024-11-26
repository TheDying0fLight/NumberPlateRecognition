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
            image_fit=ft.ImageFit.FILL,
            border_radius=3, 
            image_opacity=0.8
        )