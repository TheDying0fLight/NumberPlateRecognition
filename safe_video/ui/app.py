import flet as ft
import shutil
from typing import Dict
import os
from safe_video.number_plate_recognition import NumberPlateRecognition
from .dataclasses import Video, Image, ColorPalate

CACHE_PATH = "safe_video/upload_cache/"

DarkColors = ColorPalate(
    normal = "#1a1e26",
    light = "#232833",
    dark = "#101217",
)

class UI_App:
    def __init__(self):
        self.images: list[Image] = []
        self.current_image: int
        self.cache_path = f"safe_video/upload_cache/"
        self.page: ft.Page = None
        self.colors: ColorPalate = DarkColors
        self.image_ref = ft.Ref[ft.Container]()

    def blur_callback(self):
        npr = NumberPlateRecognition(self.image_paths[-1])
        npr.blur_image()

    def upload_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.files is None: return
        for file in file_results.files:
            path =self.cache_path + file.name
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            shutil.copy(file.path, path)
            [name, format] = file.name.split(".", 1)
            self.images.append(Image(cache_path=self.cache_path, original_path=file.path, name=name, format=format))
            self.current_image = len(self.images)-1
            
            if self.image_ref.current is None:
                img = ft.Container(ref=self.image_ref, image_src=path, image_fit=ft.ImageFit.CONTAIN, expand=True, margin=10)
                self.page.add(img)
            else:
                self.image_ref.current.image_src=path
                self.image_ref.current.update()
    
    def export_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.path is None: return
        image = self.images[self.current_image]
        export_path = file_results.path
        if '.' not in export_path:
            export_path += '.' + image.format
        shutil.move(image.get_path(), export_path)

    def add_image(self, file: ft.file_picker.FilePickerFile):
        pass

    def settings_callback(self):
        print('TODO: Settings')

    def build_page(self, page: ft.Page):
        self.page = page
        page.padding=0
        page.spacing=0
        page.bgcolor = self.colors.normal
        file_picker_open = ft.FilePicker(on_result=self.upload_callback)
        file_picker_export = ft.FilePicker(on_result=self.export_callback)
        page.overlay.append(file_picker_open)
        page.overlay.append(file_picker_export)
        page.add(
            ft.Container(ft.Row([
                ft.ElevatedButton("Open Image", on_click=lambda _: file_picker_open.pick_files(file_type=ft.FilePickerFileType.IMAGE), icon=ft.icons.FOLDER_OPEN),
                ft.ElevatedButton("Export file", on_click=lambda _: file_picker_export.save_file(initial_directory=self.images[self.current_image].original_path), icon=ft.icons.SAVE_ALT),
                ft.ElevatedButton("Blur image", on_click=lambda _: self.blur_callback()),
                ft.Row([], expand=True),
                ft.IconButton(on_click=lambda _: self.blur_callback(), icon=ft.icons.SETTINGS)
            ]), padding=10, bgcolor=self.colors.dark),
        )

    def run(self):
        ft.app(target=self.build_page)


