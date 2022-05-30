"""Parliamentary questions classification task """

import csv

import datasets

from datasets.tasks import TextClassification


_DESCRIPTION = """-"""

_CITATION = """-"""

_TRAIN_DOWNLOAD_URL = "-"
_TEST_DOWNLOAD_URL = "-"


class ParliamentaryQuestionsDataset(datasets.GeneratorBasedBuilder):
    """Parliamentary Questions TOP 9 Ministry classification dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=['Ministre de la Santé',
                                           'Ministre de l Environnement',
                                           'Ministre du Développement durable et des Infrastructures',
                                           'Ministre de la Justice',
                                           'Ministre des Finances',
                                           "Ministre de l'Education nationale",
                                           'Ministre de la Sécurité sociale',
                                           "Ministre de l'Intérieur",
                                           'Premier Ministre']),
                }
            ),
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="label",
                                   labels=['Ministre de la Santé',
                                           'Ministre de l Environnement',
                                           'Ministre du Développement durable et des Infrastructures',
                                           'Ministre de la Justice',
                                           'Ministre des Finances',
                                           "Ministre de l'Education nationale",
                                           'Ministre de la Sécurité sociale',
                                           "Ministre de l'Intérieur",
                                           'Premier Ministre'])],
        )

    def _split_generators(self, dl_manager):
        train_path = "data/bert_train.csv"
        test_path = "data/bert_test.csv"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate Semantical documents examples."""
        class_to_idx_dict = {"Ministre de la Santé": 0,
                            "Ministre de l Environnement": 1,
                            "Ministre du Développement durable et des "
                            "Infrastructures": 2,
                            "Ministre de la Justice": 3,
                            "Ministre des Finances": 4,
                            "Ministre de l'Education nationale": 5,
                            "Ministre de la Sécurité sociale": 6,
                            "Ministre de l'Intérieur": 7,
                            "Premier Ministre": 8}

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                _, text, label = row
                if label not in class_to_idx_dict.keys():
                    continue
                label = class_to_idx_dict[label]
                yield id_, {"text": text, "label": label}