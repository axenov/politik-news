# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
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

# Lint as: python3
"""De Politik-News dataset."""

from __future__ import absolute_import, division, print_function

import os

import json

import nlp



_CITATION = """
DFKI
"""

_DESCRIPTION = """
German political news dataset
"""
_URL_dataset = ""

_TITLE = "title"
_TEXT = "text"
_SOURCE = "source"
_FOCUS = "focus"
_DATE = "date"

_BIAS = 'bias'
_QUALITY = 'quality'
_CLASS = 'class'

_CONSERVATIVE_LEFT_BIAS = 'conservative_left_bias'
_CONSERVATIVE_LEFT_QUALITY = 'conservative_left_quality'
_CONSERVATIVE_LEFT_CLASS = 'conservative_left_class'

_LIBERAL_LEFT_BIAS = 'liberal_left_bias'
_LIBERAL_LEFT_QUALITY = 'liberal_left_quality'
_LIBERAL_LEFT_CLASS = 'liberal_left_class'

_LIBERAL_RIGHT_BIAS = 'liberal_right_bias'
_LIBERAL_RIGHT_QUALITY = 'liberal_right_quality'
_LIBERAL_RIGHT_CLASS = 'liberal_right_class'

_CONSERVATIVE_RIGHT_BIAS = 'conservative_right_bias'
_CONSERVATIVE_RIGHT_QUALITY = 'conservative_right_quality'
_CONSERVATIVE_RIGHT_CLASS = 'conservative_right_class'   


class PolitikNews(nlp.GeneratorBasedBuilder):

    VERSION = nlp.Version("1.0.0")

    def _info(self):
        info = nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features({
            	_TITLE: nlp.Value("string"),
            	_TEXT: nlp.Value("string"),
            	_SOURCE: nlp.Value("string"),
            	_FOCUS: nlp.Value("string"),
            	_DATE: nlp.Value("string"),
            	_BIAS: nlp.Value("float32"),
            	_QUALITY: nlp.Value("float32"),
            	_CLASS: nlp.Value("string"),
            	_CONSERVATIVE_LEFT_BIAS: nlp.Value("float32"),
            	_CONSERVATIVE_LEFT_QUALITY: nlp.Value("float32"),
            	_CONSERVATIVE_LEFT_CLASS: nlp.Value("string"),
            	_LIBERAL_LEFT_BIAS: nlp.Value("float32"),
            	_LIBERAL_LEFT_QUALITY: nlp.Value("float32"),
            	_LIBERAL_LEFT_CLASS: nlp.Value("string"),
            	_LIBERAL_RIGHT_BIAS: nlp.Value("float32"),
            	_LIBERAL_RIGHT_QUALITY: nlp.Value("float32"),
            	_LIBERAL_RIGHT_CLASS: nlp.Value("string"),
            	_CONSERVATIVE_RIGHT_BIAS: nlp.Value("float32"),
            	_CONSERVATIVE_RIGHT_QUALITY: nlp.Value("float32"),
            	_CONSERVATIVE_RIGHT_CLASS: nlp.Value("string"),
            	}),
                              
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #data_path = dl_manager.download_and_extract(_URL_dataset)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"path": "train.jsonl"},),
            nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"path":  "test.jsonl"},),
        ]


    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path) as f:
            for i, line in enumerate(f):
                elem = json.loads(line)
                yield i, {
                    _TITLE: elem['title'],
                    _TEXT: elem['text'],
                    _SOURCE: elem['source_domain'],
                    _FOCUS: elem['focus'],
                    _DATE: elem['date_publish'],
                    _BIAS: elem['bias'],
                    _QUALITY: elem['quality'],
                    _CLASS: elem['class'],
                    _CONSERVATIVE_LEFT_BIAS: elem['conservative_left_bias'],
                    _CONSERVATIVE_LEFT_QUALITY: elem['conservative_left_quality'],
                    _CONSERVATIVE_LEFT_CLASS: elem['conservative_left_class'],
                    _LIBERAL_LEFT_BIAS: elem['liberal_left_bias'],
                    _LIBERAL_LEFT_QUALITY: elem['liberal_left_quality'],
                    _LIBERAL_LEFT_CLASS: elem['liberal_left_class'],
                    _LIBERAL_RIGHT_BIAS: elem['liberal_right_bias'],
                    _LIBERAL_RIGHT_QUALITY: elem['liberal_right_quality'],
                    _LIBERAL_RIGHT_CLASS: elem['liberal_right_class'],
                    _CONSERVATIVE_RIGHT_BIAS: elem['conservative_right_bias'],
                    _CONSERVATIVE_RIGHT_QUALITY: elem['conservative_right_quality'],
                    _CONSERVATIVE_RIGHT_CLASS: elem['conservative_right_class'],
                } 
