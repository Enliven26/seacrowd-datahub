# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
"""

_DATASETNAME = "mongabay"

_DESCRIPTION = """\
Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification.
The dataset consists of 31 important topics that are commonly found in Indonesian conservation articles or general news, and each article can belong to more than one topic.
After gathering topics for each article, each article will be classified into one of author's sentiments (positive, neutral, negative) based on related topics
"""

_HOMEPAGE = "https://huggingface.co/datasets/Datasaur/mongabay-experiment"

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNLICENSE.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://huggingface.co/datasets/Datasaur/mongabay-experiment/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true",
        "validation": "https://huggingface.co/datasets/Datasaur/mongabay-experiment/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet?download=true",
        "test": "https://huggingface.co/datasets/Datasaur/mongabay-experiment/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet?download=true",
    }
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS, Tasks.TOPIC_MODELING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class Mongabay(datasets.GeneratorBasedBuilder):
    """Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []

    subset_id = "Multi-label"

    BUILDER_CONFIGS.append(
        SEACrowdConfig(
            name=f"{subset_id}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=subset_id,
        )
    )

    seacrowd_schema = f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.TOPIC_MODELING]).lower()}"

    BUILDER_CONFIGS.append(
        SEACrowdConfig(
            name=f"{subset_id}_{seacrowd_schema}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} {seacrowd_schema} schema",
            schema=f"{seacrowd_schema}",
            subset_id=subset_id,
        )
    )

    subset_id = "Sentiment-classification"

    BUILDER_CONFIGS.append(
        SEACrowdConfig(
            name=f"{subset_id}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=subset_id,
        )
    )

    seacrowd_schema = f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SENTIMENT_ANALYSIS]).lower()}"

    BUILDER_CONFIGS.append(
        SEACrowdConfig(
            name=f"{subset_id}_{seacrowd_schema}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} {seacrowd_schema} schema",
            schema=f"{seacrowd_schema}",
            subset_id=subset_id,
        )
    )

    DEFAULT_CONFIG_NAME = "Multi-label_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "filename": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "softlabel": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.TOPIC_MODELING]).lower()}":
            # TODO: fix invalid string error for train label
            features = schemas.text_features(["positif", "negatif", "netral"])

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SENTIMENT_ANALYSIS]).lower()}":
            # TODO: fix invalid string error for train label
            features = schemas.text_features(["positif", "negatif", "netral"])

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        split_generators = []

        for split in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]:
            path = dl_manager.download_and_extract(_URLS[_DATASETNAME][str(split)])

            split_generators.append(
                datasets.SplitGenerator(
                    name=str(split),
                    gen_kwargs={
                        "path": path,
                        "split": split,
                    },
                )
            )

        return split_generators

    def _generate_examples(self, path: str, split: datasets.Split) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        idx = 0

        df = pd.read_parquet(path)
        df["temp"] = df["softlabel"].apply(lambda x: str(x).split(","))

        if split == datasets.Split.TRAIN:
            if self.config.subset_id == "Sentiment-classification":
                df = df[df["temp"].apply(len) == 3]

            elif self.config.subset_id == "Multi-label":
                df = df[df["temp"].apply(len) > 3]

            else:
                raise ValueError(f"Invalid subset_id for config name: {self.config.name}")

        else:
            if self.config.subset_id == "Sentiment-classification":
                df = df[df["temp"].apply(len) == 1]

            elif self.config.subset_id == "Multi-label":
                df = df[df["temp"].apply(len) > 1]

            else:
                raise ValueError(f"Invalid subset_id for config name: {self.config.name}")

        df.drop(columns=["temp"], inplace=True)

        if self.config.schema == "source":
            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        elif self.config.schema == "seacrowd_text":
            df["id"] = df.index
            df.rename(columns={"softlabel": "label"}, inplace=True)
            df.drop(columns=["title"], inplace=True)
            df.drop(columns=["filename"], inplace=True)

            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
