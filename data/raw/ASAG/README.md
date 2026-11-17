---
pretty_name: Mohler ASAG
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
- sentence-similarity
- question-answering
size_categories:
- 1K<n<10K
tags:
- ASAG
- NLP
- Automatic Short Answer Grading
- Student Responses
- Computer Science
- Data Structure
- Educational Data
- Semantic Similarity
- Question-Answering
- Text Classification
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: instructor_answer
    dtype: string
  - name: student_answer
    dtype: string
  - name: score_grader_1
    dtype: float32
  - name: score_grader_2
    dtype: float32
  - name: score_avg
    dtype: float32
  splits:
  - name: open_ended
    num_bytes: 153600
    num_examples: 2273
  - name: close_ended
    num_bytes: 11776
    num_examples: 169
  dataset_size: 953344
configs:
- config_name: raw
  default: true
  data_files:
  - split: open_ended
    path: data/raw-oe-*
  - split: close_ended
    path: data/raw-ce-*
- config_name: cleaned
  data_files:
  - split: open_ended
    path: data/cleaned-oe-*
  - split: close_ended
    path: data/cleaned-ce-*
- config_name: parsed
  data_files:
  - split: open_ended
    path: data/parsed-oe-*
  - split: close_ended
    path: data/parsed-ce-*
- config_name: annotations
  data_files:
  - split: annotations
    path: data/annotations-*
---

<style>
  .callout {
    background-color: #cff4fc;
    border-left: 0.25rem solid #9eeaf9;
    padding: 1rem;
  }
  
  .readme-table-container table {
    font-family:monospace;
    margin: 0;
  }
</style>

# Dataset Card for "Mohler ASAG"

The **Mohler ASAG** dataset is recognized as one of the first publicly
available and widely used benchmark datasets for Automatic Short
Answer Grading (ASAG). It was first introduced by Michael Mohler and
Rada Mihalcea in 2009. An extended version of the dataset with
additional questions and corresponding student answers was released in
2011. This repository presents the 2011 dataset along with a code
snippet to extract the 2009 subset.

The dataset was collected from an introductory data structures course
at the University of North Texas. It covers 87 assessment questions in
total, including 81 open-ended and 6 closed-ended selection or
ordering questions. These questions are distributed across 10
assignments and 2 examinations. Altogether, the dataset contains 2,442
student responses, with 2,273 corresponding to open-ended questions
and 169 to closed-ended questions.

- **Authors:** Michael Mohler, Razvan Bunescu, and Rada Mihalcea.
- **Paper:** [Learning to Grade Short Answer Questions using Semantic
  Similarity Measures and Dependency Graph Alignments](https://aclanthology.org/P11-1076/)

<div class="callout">
  A curated version of the dataset is available on Hugging Face at
  <a href="https://huggingface.co/datasets/nkazi/MohlerASAG-Curated">
    <code>nkazi/MohlerASAG-Curated</code>
  </a>,
  created to improve its quality and usability for NLP research,
  particularly for LLM-based approaches.
</div>

## Known Errata

1. The 2009 paper reports 30 student answers per question for each
   assignment. In reality, assignment 1 contains 29 answers per
   question, assignment 2 contains 30 answers per question, and
   assignment 3 contains 31 answers per question.
2. The 2011 paper states that the dataset contains student answers for
   80 questions. According to the README file included with the data,
   it actually includes answers for 81 open-ended questions.

## Dataset Conversion Notebook

The Python notebook I developed to convert the Mohler ASAG dataset
from its source files into a Hugging Face Dataset is available on my
GitHub profile. It exhibits the process of parsing questions,
instructor answers, student answers, scores, and annotations from
their respective source files for each stage, correcting mojibakes in
the raw data, structuring and organizing the information, dividing and
transforming the data into subsets and splits, and exporting the final
dataset in Parquet format for the Hugging Face repository. This
demonstration ensures transparency, reproducibility, and traceability
of the conversion process.

<strong>GitHub Link:</strong>
<a href="https://github.com/nazmulkazi/ML-DL-NLP/blob/main/HF%20Dataset%20-%20Mohler%20ASAG.ipynb">
  https://github.com/nazmulkazi/ML-DL-NLP/blob/main/HF%20Dataset%20-%20Mohler%20ASAG.ipynb
</a>

## Dataset Structure and Details

The dataset underwent several processing stages, each represented as a
separate subset. The raw subset contains the original and unaltered
student answers exactly as written. In the cleaned subset, the authors
preprocessed the data by cleaning the text and tokenizing it into
sentences using the LingPipe toolkit, with sentence boundaries marked
by `<STOP>` tags. The parsed subset includes outputs from the Stanford
Dependency Parser with additional postprocessing performed by the
authors. The annotations subset contains manually annotated data.
However, only 32 student answers were randomly selected for
annotation.

The authors ignored responses to the closed-ended questions in all of
their work. Therefore, the raw, cleaned, and parsed subsets are
divided into open-ended and closed-ended splits.

Each sample in the raw, cleaned, and parsed subsets includes a unique
identifier, the question, the instructor's answer, the student's
answer, scores from two graders, and the average score. Samples in the
annotations subset contain a unique identifier and the corresponding
annotations. The unique identifiers are consistent across all subsets
and follow the format `EXX.QXX.AXX`, where each component corresponds
to its exercise (i.e., assignment), question, and answer, respectively,
and `XX` are zero-padded numbers. For consistency, reproducibility,
and traceability, the identifiers are constructed following the same
indexing scheme used by the authors, with 1-based numbering for
exercises and questions and 0-based numbering for student answers.

Exercises E01 through E10 were graded on a 0-5 scale, while E11 and
E12 were graded on a 0-10 scale. The scores for E11 and E12 were
converted to a 0-5 scale before computing the average by the authors,
so all values in the score_avg column are in the 0-5 range. Grader 1
was the course teaching assistant, and Grader 2 was Michael Mohler.

For further details, please refer to the [README](./README-Mohler.pdf)
(a formatted and styled version of the README provided by the authors)
and the associated publications.

## Student Answer Distribution

Distribution of student answers in the raw, cleaned, and parsed subsets:

<div class="readme-table-container">

|         | Q01 | Q02 | Q03 | Q04 | Q05 | Q06 | Q07 | Q08 | Q09 | Q10 | Total |
|:--------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|------:|
| **E01** |  29 |  29 |  29 |  29 |  29 |  29 |  29 |   - |   - |   - |   203 |
| **E02** |  30 |  30 |  30 |  30 |  30 |  30 |  30 |   - |   - |   - |   210 |
| **E03** |  31 |  31 |  31 |  31 |  31 |  31 |  31 |   - |   - |   - |   217 |
| **E04** |  30 |  30 |  30 |  30 |  30 |  30 |  30 |   - |   - |   - |   210 |
| **E05** |  28 |  28 |  28 |  28 |   - |   - |   - |   - |   - |   - |   112 |
| **E06** |  26 |  26 |  26 |  26 |  26 |  26 |  26 |   - |   - |   - |   182 |
| **E07** |  26 |  26 |  26 |  26 |  26 |  26 |  26 |   - |   - |   - |   182 |
| **E08** |  27 |  27 |  27 |  27 |  27 |  27 |  27 |   - |   - |   - |   189 |
| **E09** |  27 |  27 |  27 |  27 |  27 |  27 |  27 |   - |   - |   - |   189 |
| **E10** |  24 |  24 |  24 |  24 |  24 |  24 |  24 |   - |   - |   - |   168 |
| **E11** |  30 |  30 |  30 |  30 |  30 |  30 |  30 |  30 |  30 |  30 |   300 |
| **E12** |  28 |  28 |  28 |  28 |  28 |  28 |  28 |  28 |  28 |  28 |   280 |

</div>

Distribution of student answers in the annotations subset/split:

<div class="readme-table-container">

|         | Q01 | Q02 | Q03 | Q04 | Q05 | Q06 | Q07 | Total |
|:--------|----:|----:|----:|----:|----:|----:|----:|------:|
| **E01** |   3 |   3 |   3 |   3 |   2 |   1 |   1 |    16 |
| **E02** |   1 |   1 |   1 |   2 |   1 |   1 |   1 |     8 |
| **E03** |   1 |   1 |   1 |   1 |   1 |   1 |   2 |     8 |

</div>

## Code Snippets
### Extracting 2009 Dataset

Exercises 1-3 are inherited from the 2009 dataset. The following code
extracts the raw samples of the 2009 dataset from the raw subset:

```python
from datasets import load_dataset

ds = load_dataset('nkazi/MohlerASAG', name='raw', split='open_ended')
ds_2009 = ds.filter(lambda row: row['id'].split('.')[0] in ['E01', 'E02', 'E03'])
```

### Concatenating Splits

The following code creates a new dataset with rows from both
open-ended and close-ended splits from the raw subset:

```python
from datasets import load_dataset
from datasets import concatenate_datasets

ds = load_dataset('nkazi/MohlerASAG', name='raw')
ds_all = concatenate_datasets([ds['open_ended'], ds['close_ended']]).sort('id')
```

### Joining Open-Ended Raw Data with Annotations

The following code joins the annotations with their corresponding
samples from the raw subset.

```python
from datasets import load_dataset

# Load the annotations split and create a mapping
# from IDs to their annotations.
ds_ann = load_dataset('nkazi/MohlerASAG', name='annotations', split='annotations')
ann_map = {row['id']: row['annotations'] for row in ds_ann}

# Load the raw open-ended subset and keep only rows
# with IDs present in the annotations set.
ds_raw = load_dataset('nkazi/MohlerASAG', name='raw', split='open_ended') \
  .filter(lambda row: row['id'] in ann_map)

# Collect annotations in the same order as the IDs in
# the filtered raw dataset.
ann_list = [ann_map.get(row_id, None) for row_id in ds_raw['id']]

# Add an annotations column to the filtered raw dataset,
# using the annotations list and feature specification
# from the annotations subset.
ds_joined = ds_raw.add_column(
  name = 'annotations',
  column = ann_list,
  feature = ds_ann.features['annotations']
)
```

## Citation

In addition to citing **Mohler et al. (2011)**, we kindly request that
a footnote be included referencing the Hugging Face page of this dataset
([https://huggingface.co/datasets/nkazi/MohlerASAG](https://huggingface.co/datasets/nkazi/MohlerASAG))
in order to inform the community of this readily usable version.

```tex
@inproceedings{mohler2011learning,
  title = {Learning to Grade Short Answer Questions using Semantic
    Similarity Measures and Dependency Graph Alignments},
  author = {Mohler, Michael and Bunescu, Razvan and Mihalcea, Rada},
  year = 2011,
  month = jun,
  booktitle = {Proceedings of the 49th Annual Meeting of the Association
    for Computational Linguistics: Human Language Technologies},
  pages = {752--762},
  editor = {Lin, Dekang and Matsumoto, Yuji and Mihalcea, Rada},
  publisher = {Association for Computational Linguistics},
  address = {Portland, Oregon, USA},
  url = {https://aclanthology.org/P11-1076},
}
```