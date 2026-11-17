---
pretty_name: Beetle
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: reference_answer
    dtype: string
  - name: student_answer
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': correct
          '1': contradictory
          '2': partially_correct_incomplete
          '3': irrelevant
          '4': non_domain
  splits:
  - name: train
    num_examples: 3941
    num_bytes: 120274
  - name: test_ua
    num_examples: 439
    num_bytes: 21208
  - name: test_uq
    num_examples: 819
    num_bytes: 27339
  dataset_size: 168821
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test_ua
    path: data/test-ua-*
  - split: test_uq
    path: data/test-uq-*
- config_name: all_reference_answers
  data_files:
  - split: all_reference_answers
    path: data/all-reference-answers-*
---

# Dataset Card for "Beetle"

The Beetle dataset is one of the two distinct subsets within the Student Response Analysis (SRA) corpus, the other subset being the
[SciEntsBank](https://huggingface.co/datasets/nkazi/SciEntsBank) dataset. The Beetle dataset is based on transcripts generated from
students interacting with the Beetle II tutorial dialogue system [1]. The dataset includes over 5K answers to 56 questions in basic
electricity and electronics domain. The dataset features three labeling schemes: (a) 5-way, (b) 3-way, and (c) 2-way. The dataset
includes a training set and two distinct test sets: (a) Unseen Answers (`test_ua`), and (b) Unseen Questions (`test_uq`).

- **Authors:** Myroslava Dzikovska, Rodney Nielsen, Chris Brew, Claudia Leacock, Danilo Giampiccolo, Luisa Bentivogli, Peter Clark, Ido Dagan, Hoa Trang Dang
- **Paper:** [SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge](https://aclanthology.org/S13-2045)

## Loading Dataset

```python
from datasets import load_dataset
dataset = load_dataset('nkazi/beetle')
```

## Labeling Schemes

The authors released the dataset with annotations using five labels (i.e., 5-way labeling scheme) for Automated Short-Answer Grading (ASAG).
Additionally, the authors have introduced two alternative labeling schemes, namely the 3-way and 2-way schemes, both derived from the 5-way
labeling scheme designed for Recognizing Textual Entailment (RTE). In the 3-way labeling scheme, the categories "partially correct but
incomplete", "irrelevant", and "non-domain" are consolidated into a unified category labeled as "incorrect". On the other hand, the 2-way
labeling scheme simplifies the classification into a binary system where all labels except "correct" are merged under the "incorrect" category.

The `label` column in this dataset presents the 5-way labels. For 3-way and 2-way labels, use the code provided below to derive it
from the 5-way labels. After converting the labels, please verify the label distribution. A code to print the label distribution is
also given below.

### Converting 5-way to 3-way

```python
from datasets import ClassLabel

dataset = dataset.align_labels_with_mapping({'correct': 0, 'contradictory': 1, 'partially_correct_incomplete': 2, 'irrelevant': 2, 'non_domain': 2}, 'label')
dataset = dataset.cast_column('label', ClassLabel(names=['correct', 'contradictory', 'incorrect']))
```

Using `align_labels_with_mapping()`, we are mapping "partially correct but incomplete", "irrelevant", and "non-domain" to the same id. Subsequently,
we are using `cast_column()` to redefine the class labels (i.e., the label feature) where the id 2 corresponds to the "incorrect" label.

### Converting 5-way to 2-way

```python
from datasets import ClassLabel

dataset = dataset.align_labels_with_mapping({'correct': 0, 'contradictory': 1, 'partially_correct_incomplete': 1, 'irrelevant': 1, 'non_domain': 1}, 'label')
dataset = dataset.cast_column('label', ClassLabel(names=['correct', 'incorrect']))
```

In the above code, the label "correct" is mapped to 0 to maintain consistency with both the 5-way and 3-way labeling schemes. If the preference is to
represent "correct" with id 1 and "incorrect" with id 0, either adjust the label map accordingly or run the following to switch the ids:

```python
dataset = dataset.align_labels_with_mapping({'incorrect': 0, 'correct': 1}, 'label')
```

### Saving and loading 3-way and 2-way datasets

Use the following code to store the dataset with the 3-way (or 2-way) labeling scheme locally to eliminate the need to convert labels each time the dataset is loaded:

```python
dataset.save_to_disk('Beetle_3way')
```

Here, `Beetle_3way` depicts the path/directory where the dataset will be stored. Use the following code to load the dataset from the same local directory/path:

```python
from datasets import DatasetDict
dataset = DatasetDict.load_from_disk('Beetle_3way')
```

### Printing label distribution

Use the following code to print the label distribution:

```python
def print_label_dist(dataset):
    for split_name in dataset:
        print(split_name, ':')
        num_examples = 0
        for label in dataset[split_name].features['label'].names:
            count = dataset[split_name]['label'].count(dataset[split_name].features['label'].str2int(label))
            print(' ', label, ':', count)
            num_examples += count
        print('  total :', num_examples)

print_label_dist(dataset)
```

## Label Distribution

<style>
.label-dist table {display: table; width: 100%;}
.label-dist th:not(:first-child), .label-dist td:not(:first-child) {
  width: 20%;
}
</style>

<div class="label-dist">

### 5-way

Label | Train | Test UA | Test UQ
--- | --: | --: | --:
Correct | 1,665 | 176 | 344
Contradictory | 1,049 | 111 | 244
Partially correct but incomplete | 919 | 112 | 172
Irrelevant | 113 | 17 | 19
Non-domain | 195 | 23 | 40
Total | 3,941 | 439 | 819

### 3-way

Label | Train | Test UA | Test UQ
--- | --: | --: | --:
Correct | 1,665 | 176 | 344
Contradictory | 1,049 | 111 | 244
Incorrect | 1,227 | 152 | 231
Total | 3,941 | 439 | 819

### 2-way

Label | Train | Test UA | Test UQ
--- | --: | --: | --:
Correct | 1,665 | 176 | 344
Incorrect | 2,276 | 263 | 475
Total | 3,941 | 439 | 819

</div>

## Reference Answers

The Beetle dataset, initially disseminated by the authors in XML format, contained multiple reference answers for each question,
categorized under three standards: (a) best, (b) good, and (c) minimal. To align with common usage and adhere to standard coding
practices, the `reference_answer` column in this instance of the dataset is populated with the first best reference answer per
question. The complete array of reference answers is accessible in the `all_reference_answers` split. This split is excluded from
the default configuration and will not be loaded by the `load_dataset` method by default. Utilize the following code to load/access
this split:

```python
from datasets import load_dataset
dataset = load_dataset("nkazi/Beetle", name='all_reference_answers')
```

Please note that in this code, the `name` parameter refers to the configuration containing the `all_reference_answers` split,
rather than specifying a split name.

## Citation

Please consider adding a **footnote** linking to this dataset page (e.g., `Beetle\footnote{https://huggingface.co/datasets/nkazi/Beetle}` in LaTeX)
when first mentioning the dataset in your paper, alongside citing the authors/paper. This will promote the availability of this dataset on
Hugging Face and make it more accessible to researchers, given that the original repository is no longer available.

```tex
@inproceedings{dzikovska2013semeval,
  title = {{S}em{E}val-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge},
  author = {Dzikovska, Myroslava and Nielsen, Rodney and Brew, Chris and Leacock, Claudia and Giampiccolo, Danilo and Bentivogli, Luisa and Clark, Peter and Dagan, Ido and Dang, Hoa Trang},
  year = 2013,
  month = jun,
  booktitle = {Second Joint Conference on Lexical and Computational Semantics ({SEM}), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation ({S}em{E}val 2013)},
  editor = {Manandhar, Suresh  and Yuret, Deniz}
  publisher = {Association for Computational Linguistics},
  address = {Atlanta, Georgia, USA},
  pages = {263--274},
  url = {https://aclanthology.org/S13-2045},
}
```

## References
1. Myroslava O. Dzikovska, Johanna D. Moore, Natalie Steinhauser, Gwendolyn Campbell, Elaine Farrow, and Charles B. Callaway. 2010.
Beetle II: A system for tutoring and computational linguistics experimentation. In *Proc. of ACL 2010 System Demonstrations*, pages 13–18.