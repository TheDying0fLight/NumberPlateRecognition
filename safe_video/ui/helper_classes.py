import os
import shutil
import re
from typing import Dict

from .dataclasses import Video, Image, ColorPalette

class FileManger(Dict[str, Image]):
    def __init__(self, cache_path, **kwargs):
        self.cache_path = cache_path
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        super().__init__(**kwargs)

    def load_cached(self):
        names = []
        for filename in os.listdir(self.cache_path):
            name, counter, fmt = re.findall("(.*)_(\d)+\.(.*)$", filename)[0]
            path_name = name + '_' + counter
            self.__setitem__(path_name, Image(cache_path=self.cache_path, name=path_name, orig_name=name, format=fmt))
            names.append(path_name)
        return names

    def upload_image(self, old_path: str, filename: str):
        """Uploads the image and inserts it into the dictionary

        Args:
            old_path (str): Path the image should be loaded from
            name (str): name of the file
            format (str): format (e.g. png)
        """
        name, fmt = re.findall("(.*)\.(.*)$", filename)[0]
        new_path = self.cache_path + name + "{}." + fmt
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        counter = 0
        new_path = str(self.cache_path + name + "_{}." + fmt)
        while os.path.isfile(new_path.format(counter)):
            counter += 1
        shutil.copy(old_path, new_path.format(counter))
        path_name = name + '_' + str(counter)
        self.__setitem__(path_name, Image(cache_path=self.cache_path, name=path_name, orig_name=name, format=fmt))
        return path_name

    def export_image(self, name, export_path):
        img = self.__getitem__(name)
        if '.' not in export_path:
            export_path += '.' + img.format
        shutil.copy(img.get_path(), export_path)
        img.saved = True

    def __delitem__(self, name):
        img = self.__getitem__(name)
        os.remove(img.get_path())
        super().__delitem__(name)

