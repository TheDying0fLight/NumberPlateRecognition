import flet as ft
import shutil
from typing import Dict
import os
from safe_video.number_plate_recognition import NumberPlateRecognition
from .dataclasses import Video, Image, ColorPalette
from .components import PreviewImage, AlertSaveWindow

CACHE_PATH = "safe_video/upload_cache/"

DarkColors = ColorPalette(
    normal = "#1a1e26",
    light = "#232833",
    dark = "#101217",
)

class UI_App:
    def __init__(self):
        self.images: Dict[str, Image] = {}
        self.cache_path = CACHE_PATH
        self.page: ft.Page = None
        self.colors: ColorPalette = DarkColors
        self.image_ref = ft.Ref[ft.Container]()
        self.preview_bar_ref = ft.Ref[ft.Column]()
        self.selected_img: str = None
        self.selected = set()
        self.npr = NumberPlateRecognition()
        self.file_picker_open = ft.FilePicker(on_result=self.upload_callback)
        self.file_picker_export = ft.FilePicker(on_result=self.export_callback)

    def blur_callback(self):
        self.npr.blur_image()

    def upload_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.files is None or len(file_results.files) == 0: return
        images = []
        for file in file_results.files:
            path = self.cache_path + file.name
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            shutil.copy(file.path, path)
            [name, format] = file.name.split(".", 1)
            images.append(Image(cache_path=self.cache_path, name=name, format=format))
        self.load_images(images)

    def load_images(self, images: list[Image]):
        if len(images) == 0: return
        for img in images:
            self.images[img.name] = img
            img.preview_ref = PreviewImage(img.name, img.get_path(), self.switch_image_callback)
            self.preview_bar_ref.current.controls.append(img.preview_ref)
        self.switch_image(images[-1].name)
        self.page.update()

    def switch_image(self, name):
        if self.selected_img is not None:
            if name == self.selected_img: return
            self.images[self.selected_img].selected(False)
        if name not in self.images:
            self.selected_img = None
            self.image_ref.current.content = None
        self.selected_img = name
        img = self.images[name]
        img.selected(True)
        self.image_ref.current.content = ft.Image(img.get_path(), fit=ft.ImageFit.CONTAIN)
        self.page.update()

    def switch_image_callback(self, info: ft.ControlEvent):
        name = info.control.key
        self.switch_image(name)

    def export_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.path is None: return
        img = self.images[self.selected_img]
        export_path = file_results.path
        if '.' not in export_path:
            export_path += '.' + img.format
        shutil.copy(img.get_path(), export_path)
        img.saved = True
        if img.closed:
            self.close_image(img.name)

    def close_image(self, name):
        img = self.images[name]
        os.remove(img.get_path())
        self.preview_bar_ref.current.controls = [c for c in self.preview_bar_ref.current.controls if c.key != img.name]
        del img
        self.switch_image(list(self.images.keys())[0] if len(self.images) >= 1 else '')
        self.page.update()

    def close_callback(self, info: ft.ControlEvent):
        if self.selected_img is None: return
        img = self.images[self.selected_img]
        img.closed = True
        if img.saved:
            self.close_image(img.name)
        else:
            self.page.open(AlertSaveWindow(
                save_callback=lambda: self.file_picker_export.save_file(file_name=img.name),
                close_callback=lambda: self.close_image(img.name)
            ))

    def settings_callback(self, info: ft.ControlEvent):
        print('TODO: Settings')

    def build_page(self, page: ft.Page):
        self.page = page
        page.padding = 0
        page.spacing = 0
        page.bgcolor = self.colors.light
        page.on_keyboard_event = lambda e: print(e)
        page.overlay.append(self.file_picker_open)
        page.overlay.append(self.file_picker_export)
        page.add(
            ft.Container(ft.Row([
                ft.Container(content=ft.IconButton(ft.icons.BLUR_ON, focus_color=self.colors.dark), width=50),
                ft.ElevatedButton("Open Image", on_click=lambda _: self.file_picker_open.pick_files(file_type=ft.FilePickerFileType.IMAGE, allow_multiple=True), icon=ft.icons.FOLDER_OPEN),
                ft.ElevatedButton("Export file", on_click=lambda _: self.file_picker_export.save_file(file_name=self.images[self.selected_img].name), icon=ft.icons.SAVE_ALT),
                ft.ElevatedButton("Close file", on_click=self.close_callback, icon=ft.icons.DELETE),
                ft.ElevatedButton("Blur all", on_click=lambda _: self.blur_callback(), icon=ft.icons.PLAY_ARROW),
                ft.Row([], expand=True),
                ft.IconButton(on_click=self.settings_callback, icon=ft.icons.SETTINGS)
            ]), padding=10, bgcolor=self.colors.dark),
            ft.Row([
                ft.Container(ft.Column([], expand=True, spacing=10, ref=self.preview_bar_ref), bgcolor=self.colors.normal, padding=10, width=70),
                ft.Container(ref=self.image_ref, expand=True, image_fit=ft.ImageFit.CONTAIN, margin=10),
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
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        images = []
        for filename in os.listdir(self.cache_path):
            [name, format] = filename.split(".", 1)
            images.append(Image(cache_path=self.cache_path, name=name, format=format))
        self.load_images(images)

    def run(self):
        ft.app(target=self.build_page)