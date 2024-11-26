import flet as ft
import shutil
from typing import Dict
import os
from safe_video.number_plate_recognition import NumberPlateRecognition
from .dataclasses import Video, Image, ColorPalate
from .components import PreviewImage

CACHE_PATH = "safe_video/upload_cache/"

DarkColors = ColorPalate(
    normal = "#1a1e26",
    light = "#232833",
    dark = "#101217",
)

class UI_App:
    def __init__(self):
        self.images: dict[Image] = {}
        self.cache_path = f"safe_video/upload_cache/"
        self.page: ft.Page = None
        self.colors: ColorPalate = DarkColors
        self.image_ref = ft.Ref[ft.Row]()
        self.preview_bar_ref = ft.Ref[ft.Column]()
        self.current_image_key: str
        self.npr = NumberPlateRecognition()

    def blur_callback(self):
        self.npr.blur_image()

    def upload_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.files is None or len(file_results.files) == 0: return
        for file in file_results.files:
            path =self.cache_path + file.name
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            shutil.copy(file.path, path)
            [name, format] = file.name.split(".", 1)
            self.images[name] = Image(cache_path=self.cache_path, original_path=file.path, name=name, format=format)
            self.current_image = name
            
            self.preview_bar_ref.current.controls.append(PreviewImage(name, path, self.switch_image))
            self.preview_bar_ref.current.update()

        self.current_image_key = name
        self.image_ref.current.controls = [ft.Container(image_src=path, image_fit=ft.ImageFit.CONTAIN, expand=True, margin=10)]
        self.image_ref.current.update()
    
    def switch_image(self, info: ft.ControlEvent):
        self.current_image_key = info.control.key
        img = self.images[self.current_image_key]
        self.image_ref.current.controls = [ft.Container(image_src=img.get_path(), image_fit=ft.ImageFit.CONTAIN, expand=True, margin=10)]
        self.image_ref.current.update()

    def export_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.path is None: return
        img = self.images[self.current_image_key]
        self.preview_bar_ref.current.controls = [c for c in self.preview_bar_ref.current.controls if c.key != img.name]
        self.preview_bar_ref.current.update()
        export_path = file_results.path
        if '.' not in export_path:
            export_path += '.' + img.format
        shutil.move(img.get_path(), export_path)

    def settings_callback(self):
        print('TODO: Settings')

    def build_page(self, page: ft.Page):
        self.page = page
        page.padding=0
        page.spacing=0
        page.bgcolor = self.colors.light
        file_picker_open = ft.FilePicker(on_result=self.upload_callback)
        file_picker_export = ft.FilePicker(on_result=self.export_callback)
        page.overlay.append(file_picker_open)
        page.overlay.append(file_picker_export)
        page.add(
            ft.Container(ft.Row([
                ft.Container(content=ft.IconButton(ft.icons.BLUR_ON, focus_color=self.colors.dark), width=50),
                ft.ElevatedButton("Open Image", on_click=lambda _: file_picker_open.pick_files(file_type=ft.FilePickerFileType.IMAGE, allow_multiple=True), icon=ft.icons.FOLDER_OPEN),
                ft.ElevatedButton("Export file", on_click=lambda _: file_picker_export.save_file(file_name=self.images[self.current_image_key].name), icon=ft.icons.SAVE_ALT),
                ft.ElevatedButton("Blur all", on_click=lambda _: self.blur_callback(), icon=ft.icons.PLAY_ARROW),
                ft.Row([], expand=True),
                ft.IconButton(on_click=lambda _: self.blur_callback(), icon=ft.icons.SETTINGS)
            ]), padding=10, bgcolor=self.colors.dark),
            ft.Row([
                ft.Container(ft.Column([], expand=True, spacing=10, ref=self.preview_bar_ref), bgcolor=self.colors.normal, padding=10, width=70),
                ft.Row([], ref=self.image_ref, expand=True),
                ft.Container(ft.Column([
                    ft.ExpansionTile(
                        title=ft.Text("Number Plates"),
                        leading=ft.Checkbox(),
                        shape=ft.StadiumBorder(),
                        controls=[ft.Text("options")]
                        ),
                    ft.ExpansionTile(
                        title=ft.Text("Faces"),
                        leading=ft.Checkbox(),
                        shape=ft.StadiumBorder(),
                        controls=[ft.Text("options")]
                        ),
                ], expand=True), bgcolor=self.colors.normal, width=300, expand=0.5, alignment=ft.alignment.top_left),
            ], expand=True),
              
        )
        self.image_ref.current.update()
        self.preview_bar_ref.current.update()

    def run(self):
        ft.app(target=self.build_page)


