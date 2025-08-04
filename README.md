# Cascade Classifier
<img src="https://github.com/bnrp/CascadeClassifier/blob/main/Output%20Images/Figures/ModelDiagram.png" alt="Cascade Head Diagram">

## Steps to install and start running

1. Clone this repo to the desired spot. We assume that python installation and the like is complete.

2. Download the NABirds Dataset

2. Create a venv with the required classes in `requirements.txt`:
```
virtualenv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

3. **Download our trained models** [here](https://drive.google.com/drive/folders/19bi8WGWNTyf1CIitF1LVrOUUPQsh9q4A?usp=drive_link) and place them in `./Saved_Models`. You will need to modify the model loading lines in any files you wish to run.

## Files
- `/Citations/`: contains bibtex files used for bibliography.
- `/labels/`: contains pre-generated files that allow easy label and hierarchy access. Additional files include t-SNE embeddings of the test set and class weights for CBCE losses.
- `/Model/`: contains `CascadeHead.py` containing the code for the Cascade Classifier Head.
- `/nabirds-data/`: contains `nabirds.md`, which gives instructions for downloading the NABirds dataset.
- `/Output Images/`: contains many of the figures and image outputs generated during the project.
- `/Pretrained/`: put any pre-trained models for fine-tuning here.
- `/Results/`: contains confusion matrices for each of the three model outputs.
- `/Saved_Models/`: folder in which all trained models should go for analysis.
- `/utils/`: contains old notebooks, files used to generate class hierarchy, utility files for loading dataset, and misc utilities.
- `cascade_head.ipynb`: notebook used for all training of models.
- `Evaluation.ipynb`: notebook used for finding model accuracies, analyzing results, etc.
- `GradCAM.ipynb`: notebook with GradCAM implementation for developed models, used for generating figures and visualizations.
- `README.md`: this file.
- `requirements.txt`: contains all libraries needed by python to run this code.
- `tsne.py`: allows for quicker running of t-SNE plot generation and class-subset t-SNE embeddings.
