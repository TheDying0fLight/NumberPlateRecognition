# NumberPlateRecognition

This projects aims to create a program to make images and videos safe to upload under strict data protection laws.

Start the UI by installing the conda environment and executing the module:
```
conda env create --name NumberPlateRecognition --file=environment.yml
python -m safe_video
```
Useful examples for working with the package can be found in the Jupyter Notebook [`test.ipynb`](https://github.com/Cari1111/NumberPlateRecognition/blob/main/test.ipynb)

## UI
<img height="250" alt="main" src="https://github.com/user-attachments/assets/618b3891-0fab-438b-bbf8-dce026768c2c" />
<img height="250" alt="add-pipeline" src="https://github.com/user-attachments/assets/416d2146-8c34-4301-9901-f44c203a2aa8" />
<img height="250" alt="add-models" src="https://github.com/user-attachments/assets/2ed80399-473f-4824-a36e-3787c0f2ba97" />

## Models

- YOLO standard model: is imported automatically
- License_Plate model: is imported automatically (from [`/models/first10ktrain/weights/licensePlate.pt`](https://github.com/Cari1111/NumberPlateRecognition/blob/main/models/first10ktrain/weights/licensePlate.pt))
- face model: best performing model can be imported from [`/models/face_recognition/weights/best.pt`](https://github.com/Cari1111/NumberPlateRecognition/blob/main/models/face_recognition/weights/best.pt)
- text recognition model: best performing model can be imported from [`/models/text/weights/text.pt`](https://github.com/Cari1111/NumberPlateRecognition/blob/main/models/text/weights/text.pt)

# Developer Information

## Rules

- Always create a new brach for a new feature. Use the first letter of your name as the start of the branch name (e.g. c-topic)
- Use typing for all functions and classes

## Conda Environment

All the dependencies for this project are saved in the environment.yml file for a conda environment. Use
```
conda env create --name NumberPlateRecognition --file=environment.yml
```
to create the conda environment. Use `ctrl+shift+p` and search for `python:Select Interpreter` to select the environment as interpreter in vs code.

When installing new packages create a new file with:
```
conda env export > environment.yml
```
To update your conda environment with the new file use:
```
conda env update --file environment.yml --prune
```

# Further Resources

## Datasets for potential creation of additional models
- [UltralyticsYOLODocs](https://docs.ultralytics.com/datasets/detect/#usage)
- [PapersWithCode](https://paperswithcode.com/datasets?task=object-detection&mod=images&page=1)
- [Kaggle](https://www.kaggle.com/datasets?tags=13207-Computer+Vision)

### Specific Topics
#### Faces
- [Face-Detection-Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)
- <https://universe.roboflow.com/browse/person/face>

#### Number Plates
- <https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4>
- <https://universe.roboflow.com/browse/transportation/anpr>

#### Numbers
- <https://universe.roboflow.com/browse/numbers>
