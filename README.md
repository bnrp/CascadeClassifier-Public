# Cascade Classifier
<img src="https://github.com/bnrp/CascadeClassifier-Public/blob/main/Output%20Images/Figures/ModelDiagram.png" alt="Cascade Head Diagram">

## Steps to install and start running

1. Clone this repo to the desired spot. We assume that python installation is complete.

2. Download the NABirds Dataset

2. Create a venv with the required classes in `requirements.txt`:
```
virtualenv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

3. **Download our trained models** here (not publicly available) and place them in `./Saved_Models`. You will need to modify the model loading lines in any files you wish to run.

## Files
- `/Citations/`: contains bibtex files used for bibliography.
- `/labels/`: contains pre-generated files that allow easy label and hierarchy access. Additional files include t-SNE embeddings of the test set and class weights for CBCE losses.
- `/Model/`: contains `CascadeHead.py` containing the code for the Cascade Classifier Head.
- `/nabirds-data/`: expected location of NABirds dataset.
- `/Output Images/`: contains many of the figures and image outputs generated during the project.
- `/Pretrained/`: put any pre-trained models for fine-tuning here.
- `/Results/`: contains confusion matrices for each of the three model outputs.
- `/Saved_Models/`: folder in which all trained models should go for analysis.
- `/utils/`: contains old notebooks, files used to generate class hierarchy, utility files for loading dataset, and misc utilities.
- `README.md`: this file.
- `MSA_CC.py`: additional attention structure experiments.
- `load.py`: utilities for quickly loading NABirds dataset.
- `loadCUB.py`: utilities for loading CUB200 dataset (not recommended with pretrained models due to overlap with ImageNet).
- `train.py`: file containing training functions.
- `requirements.txt`: contains all libraries needed by python to run this code.
- `tsne.py`: allows for quicker running of t-SNE plot generation and class-subset t-SNE embeddings.
