This project predicts movie and TV show genres using a multi-label machine learning pipeline.

The project structure modularizes the pipeline into multiple files.

No pickle or saved models are used. Every time the user runs the code the training happens again.

The dataset should be placed inside the data folder.

Project structure:

data/
    tv-shows.csv

src/
    data_loader.py
    preprocessing.py
    feature_engineering.py
    model_training.py
    evaluation.py

main.py

To run the project:

python main.py

While executing the pipeline prints clear messages explaining:

- Dataset loading
- Genre preprocessing
- Feature engineering
- Training process
- Hyperparameter tuning
- Model evaluation

The models trained are:

Support Vector Machine (LinearSVC)
Logistic Regression

Both models are trained using GridSearchCV for hyperparameter tuning.

The pipeline uses:

TF-IDF features for title and description
Metadata features including release year, duration, rating, type and platform
Multi-label binarization for genres
Class balancing