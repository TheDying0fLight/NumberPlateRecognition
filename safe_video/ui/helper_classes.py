import os
import shutil
import re
import PIL
import PIL.Image
import warnings
import cv2
import flet as ft
from matplotlib import pyplot as plt
import numpy as np

from .dataclasses import Image, Video, Media, FileVersion, FileVersionTemplate, ColorPalette, Version
from .components import ModelTile
from safe_video.number_plate_recognition import ObjectDetection, Censor, apply_censorship
from ultralytics.engine.results import Results

class ModelManager():
    def __init__(self, bounding_box_func):
        self.detection: ObjectDetection = ObjectDetection()
        self.cls = self.detection.get_classes()[0:2]
        self.active: dict[str, bool] = {c: True for c in self.cls}
        self.results: dict[str, dict[str, Results]] = dict() # dict[cls_id][img_id]
        self.bounding_box_func = bounding_box_func

    def toggle_active(self, cls_id):
        self.active[cls_id] = not self.active[cls_id]

    def get_bounding_box_fig(self, cls_id, img: Image):
        plot = self.analyze_or_from_cache(cls_id, img).plot()
        hight, length = np.shape(plot)[0:2]
        scale = 10/min(length, hight)
        with warnings.catch_warnings(action="ignore"):
            fig = plt.figure(frameon=False, figsize=(length*scale,hight*scale))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(plot)
        return fig

    def get_blurred_img(self, cls_id, img: Image):
        img_loaded = cv2.imread(img.get_path(Version.ORIG))[:, :, ::-1]
        censored_img = apply_censorship(img_loaded, self.analyze_or_from_cache(cls_id, img), Censor.blur)
        return censored_img

    def analyze_or_from_cache(self, cls_id, img: Image) -> Results:
        if cls_id not in self.results:
            self.results[cls_id] = dict()
        if not img.id in self.results[cls_id]:
            img_loaded = cv2.imread(img.get_path(Version.ORIG))
            img_loaded = img_loaded[:, :, ::-1]
            self.results[cls_id][img.id] = self.detection.process_image(img_loaded, cls_id, conf_thresh=0.25)[0]
        return self.results[cls_id][img.id]



class FileManger(dict[str, Media]):
    def __init__(self, colors: ColorPalette):
        self.CACHE_PATH = "safe_video/upload_cache/"
        self.colors = colors
        self.ORIGINAL_TEMPLATE = FileVersionTemplate(name='original')
        self.PREVIEW_TEMPLATE_IMG = FileVersionTemplate(name='preview', fmt='webp', max_size=1000)
        self.PREVIEW_TEMPLATE_VID = FileVersionTemplate(name='preview', fmt='mp4', max_size=1000)
        self.ICON_TEMPLATE = FileVersionTemplate(name='icon', fmt='webp', max_size=100)
        self.IMAGE_FMTS = ['png', 'jpg', 'jpeg']
        self.VIDEO_FMTS = ['mp4']
        if not os.path.exists(self.CACHE_PATH):
            os.makedirs(self.CACHE_PATH)
        super().__init__()

    def load_cached(self) -> list[str]:
        """Loads all the files of the cache folder into the dict

        Returns:
            list[str]: List of the keys of the inserted files
        """
        ids = []
        for directory in os.listdir(self.CACHE_PATH):
            dir_path = f'{self.CACHE_PATH}/{directory}'
            if not os.path.isdir(dir_path): continue
            name, counter = re.findall("(.*)_(\d)+$", directory)[0]
            orig_fmt, preview_fmt, icon_fmt = '', '', ''
            for file in os.listdir(dir_path):
                file_name, fmt = re.findall("(.*)\.(.*)$", file)[0]
                if file_name == self.ORIGINAL_TEMPLATE.name:
                    orig_fmt = fmt
                elif file_name == self.PREVIEW_TEMPLATE_IMG.name:
                    preview_fmt = fmt
                elif file_name == self.ICON_TEMPLATE.name:
                    icon_fmt = fmt
                else:
                    print(f'Found unexpected file {file} in directory {directory} in cache. TODO: Handle this')
            if orig_fmt != '' and preview_fmt != '' and directory not in self:
                media = Media(id=directory, cache_path=self.CACHE_PATH, name=name)
                media.set_file(Version.ORIG, FileVersion(self.ORIGINAL_TEMPLATE.name, fmt=orig_fmt))
                media.set_file(Version.PREVIEW, FileVersion(self.PREVIEW_TEMPLATE_IMG.name, fmt=preview_fmt))
                media.set_file(Version.ICON, FileVersion(self.ICON_TEMPLATE.name, fmt=icon_fmt))
                ids.append(directory)
                if orig_fmt in self.IMAGE_FMTS:
                    self.__setitem__(directory, Image(media))
                if orig_fmt in self.VIDEO_FMTS:
                    self.__setitem__(directory, Video(media, self.__get_aspect_ratio(media.get_path(Version.ORIG))))
            else:
                print(f'Found directory {directory} with not enough files to be valid in cache. TODO: Handle this')
        return ids

    def upload_media(self, old_path: str, filename: str) -> str:
        """Uploads the image and inserts it into the dictionary

        Args:
            old_path (str): Path the image should be loaded from
            name (str): name of the file with format (e.g. img.png)
        Returns:
            str: returns the key where the image can be found
        """
        name, fmt = re.findall("(.*)\.(.*)$", filename)[0]
        if not os.path.exists(self.CACHE_PATH): # check if cache folder exists
            os.makedirs(self.CACHE_PATH)
        counter = 0
        new_folder = str(self.CACHE_PATH + name + "_{}")
        while os.path.isdir(new_folder.format(counter)):
            counter += 1
        os.makedirs(new_folder.format(counter))
        media = Media(id=f'{name}_{counter}', cache_path=self.CACHE_PATH, name=name)
        media.set_file(Version.ORIG, FileVersion(self.ORIGINAL_TEMPLATE.name, fmt=fmt))
        media.set_file(Version.PREVIEW, FileVersion(self.PREVIEW_TEMPLATE_IMG.name, fmt=self.PREVIEW_TEMPLATE_IMG.fmt))
        media.set_file(Version.ICON, FileVersion(self.ICON_TEMPLATE.name, fmt=self.ICON_TEMPLATE.fmt))
        shutil.copy(old_path, media.get_path(Version.ORIG))
        if fmt in self.IMAGE_FMTS:
            self.create_new_version_of_image(media.get_path(Version.ORIG), media.get_path(Version.PREVIEW), self.PREVIEW_TEMPLATE_IMG)
            self.create_new_version_of_image(media.get_path(Version.ORIG), media.get_path(Version.ICON), self.ICON_TEMPLATE)
            self.__setitem__(media.id, Image(media))
        elif fmt in self.VIDEO_FMTS:
            #media.preview_file.fmt = media.orig_file.fmt # TODO: change this
            self.__create_preview_and_icon_from_video(orig_path=media.get_path(Version.ORIG), preview_path=media.get_path(Version.PREVIEW), icon_path=media.get_path(Version.ICON))
            self.__setitem__(media.id, Video(media, self.__get_aspect_ratio(media.get_path(Version.ORIG))))
        return media.id

    def create_new_version_of_image(self, orig_path: str, new_path: str, template: FileVersionTemplate):
        img = PIL.Image.open(orig_path)
        width, height = img.size
        if max(width, height) > template.max_size:
            scale = template.max_size/max(width, height)
            img = img.resize((int(width*scale), int(height*scale)))
        img.save(new_path, optimize=True, quality=90)

    def create_blur_imgs(self, id, blur_result):
        img = self.__getitem__(id)
        censored_img = PIL.Image.fromarray(blur_result)
        img.set_file(Version.BLUR_ORIG, FileVersion('blur', img.files[Version.ORIG].fmt))
        censored_img.save(img.get_path(Version.BLUR_ORIG))
        img.set_file(Version.BLUR_PREVIEW, FileVersion('blur_preview', self.PREVIEW_TEMPLATE_IMG.fmt))
        self.create_new_version_of_image(img.get_path(Version.BLUR_ORIG), img.get_path(Version.BLUR_PREVIEW), self.PREVIEW_TEMPLATE_IMG)
        img.current_preview = Version.BLUR_PREVIEW

    def __create_preview_and_icon_from_video(self, orig_path: str, preview_path: str, icon_path: str):
        shutil.copy(orig_path, preview_path)
        video = cv2.VideoCapture(orig_path)
        _, image = video.read()
        cv2.imwrite(icon_path, image)
        img = PIL.Image.open(icon_path)
        width, height = img.size
        if max(width, height) > self.PREVIEW_TEMPLATE_IMG.max_size:
            scale = self.PREVIEW_TEMPLATE_IMG.max_size/max(width, height)
            img = img.resize((int(width*scale), int(height*scale)))
        img.save(icon_path, optimize=True, quality=90)

    def __get_aspect_ratio(self, path):
        video = cv2.VideoCapture(path)
        _, image = video.read()
        height, width, _ = image.shape
        return(width/height)


    def export_image(self, id: str, export_path: str):
        img = self.__getitem__(id)
        newest_version = Version.ORIG if img.files[Version.BLUR_ORIG] is None else Version.BLUR_ORIG
        if '.' not in export_path:
            export_path += '.' + img.files[newest_version].fmt
        shutil.copy(img.get_path(newest_version), export_path)
        img.saved = True

    def __delitem__(self, id: str):
        img = self.__getitem__(id)
        dir = self.CACHE_PATH + img.id
        for file in os.listdir(dir):
            os.remove(dir + '/' + file)
        os.rmdir(dir)
        super().__delitem__(id)