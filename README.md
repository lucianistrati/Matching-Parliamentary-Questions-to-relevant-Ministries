## Parliamentary Questions - Ministry Classification

### Summary

This repository documents a project focused on classifying parliamentary questions according to the relevant ministries. It explores various approaches, including classical machine learning models and deep learning methods like CamemBERT and Fasttext.

### Installation

To set up the project environment, install the required dependencies by running:

```
pip install -r requirements.txt
```

### How to Run

1. For classical machine learning models and the Fasttext model, execute the following command:
   ```
   python main.py
   ```

2. For the CamemBERT model, run the cells in the `camembert_model.ipynb` notebook.

### Repository Files

- **bert_dataset.py**: Contains code related to processing the dataset for the BERT model.
- **bert_dataset.py.lock**: Lock file for the dependencies used in the BERT dataset module.
- **camembert_model.ipynb**: Jupyter notebook implementing the CamemBERT model.
- **main.py**: Main Python script for running classical machine learning models and the Fasttext model.
- **models.py**: Module defining the machine learning models used in the project.

### Production Deployment

For production deployment, consider the following:

- Inference should be deployed in a scalable cloud environment, such as Amazon Web Services (AWS) or Microsoft Azure.
- Create an Anaconda Environment with all necessary dependencies and compatible versions. Docker can be used for platform-independent deployment by defining installation and setup commands in a Dockerfile.

### Further Work

The project suggests several areas for further exploration and improvement:

- Experiment with other word embeddings for French, such as FlauBERT.
- Explore alternative vectorization approaches, including TF-IDF vectorization at the character or word level.
- Investigate lexicon-based approaches using words frequently used in questions to specific ministries.
- Explore more aggressive preprocessing techniques for text, particularly with stop-words.



