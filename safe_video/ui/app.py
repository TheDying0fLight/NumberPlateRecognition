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
        self.image_paths = []
        self.page: ft.Page = None
        self.columns = ft.Ref[ft.Column]()
        self.colors: ColorPalate = DarkColors

    def blur_callback(self):
        npr = NumberPlateRecognition(self.image_paths[-1])
        npr.blur_image()

    def upload_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.files is None: return
        for file in file_results.files:
            destination_path = f"safe_video/upload_cache/"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            shutil.copy(file.path, destination_path + file.name)
            img = ft.Container(image_src=destination_path + file.name, image_fit=ft.ImageFit.CONTAIN, expand=True, bgcolor='#1c2130')
            self.image_paths.append(destination_path + file.name)
            #self.current_img.current.image_src = destination_path + file.name
            self.columns.current.controls.append(img)
            #self.page.add(img)
            self.page.update()

    def build_page(self, page: ft.Page):
        self.page = page
        page.padding=0
        page.bgcolor = self.colors.normal
        Mypick = ft.FilePicker(on_result=self.upload_callback)
        page.overlay.append(Mypick)
        page.add(
            ft.Column([
                ft.Container(ft.Row([
                    ft.ElevatedButton("Open file", on_click=lambda _: Mypick.pick_files()),
                    ft.ElevatedButton("Export file", on_click=lambda _: Mypick.pick_files()),
                    ft.ElevatedButton("Blur image", on_click=lambda _: self.blur_callback())
                ]), padding=10, bgcolor=self.colors.dark),
                #ft.Container(content=ft.Text("a"), expand=True, bgcolor='#1c2130'),
                ft.Container(ft.Row([
                    ft.ElevatedButton("Open file", on_click=lambda _: Mypick.pick_files()),
                    ft.ElevatedButton("Export file", on_click=lambda _: Mypick.pick_files()),
                    ft.ElevatedButton("Blur image", on_click=lambda _: self.blur_callback())
                ]), padding=10)
            ], ref=self.columns)
        )

    def run(self):
        ft.app(target=self.build_page)


