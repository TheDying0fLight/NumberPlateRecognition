class NumberPlateRecognition():

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_bounding_boxes(self):
        raise NotImplementedError('bounding_boxes not implemented')

    def blur_image(self):
        print(f'Image at {self.file_path} has to be blurred')