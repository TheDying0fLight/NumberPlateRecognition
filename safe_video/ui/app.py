import flet as ft
import shutil
from safe_video.number_plate_recognition import NumberPlateRecognition

class UI_App:
    def __init__(self):
        self.images = []
        self.page: ft.Page = None

    def blur_callback(self):
        npr = NumberPlateRecognition(self.images[-1])
        npr.blur_image()

    def upload_callback(self, file_results: ft.FilePickerResultEvent):
        for file in file_results.files:
            destination_path = f"safe_video/upload_cache/{file.name}"
            shutil.copy(file.path, destination_path)
            img = ft.Image(src=destination_path, width=100, height=100, fit=ft.ImageFit.CONTAIN)
            self.images.append(destination_path)
            self.page.add(img)

    def build_page(self, page: ft.Page):
        self.page = page
        Mypick = ft.FilePicker(on_result=self.upload_callback)
        page.overlay.append(Mypick)
        page.add(
            ft.Row([
                ft.ElevatedButton("Insert file", on_click=lambda _: Mypick.pick_files()),
                ft.ElevatedButton("Blur image", on_click=lambda _: self.blur_callback())
                ])
        )

    def run(self):
        ft.app(target=self.build_page)
