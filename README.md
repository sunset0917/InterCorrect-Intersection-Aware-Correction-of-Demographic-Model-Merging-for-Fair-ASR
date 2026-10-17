# InterCorrect: Intersection Aware Correction of Demographic Model Merging for Fair ASR

This repository contains the source code, scripts, and model checkpoints for our research on combining model merging techniques and intersection corrections in Automatic Speech Recognition (ASR).

The preprint detailing this work is available on arXiv: [2610.08604](https://arxiv.org/abs/2610.08604).

---

## Directory Structure

```text
├── InterCorrect_scripts/
├── merging_scripts/
├── SLAM-LLM/
└── models/

```

### 1. `InterCorrect_scripts/`

Contains the implementation scripts used to compute intersections and perform the "corrected" model modeling workflow as described in our paper.

### 2. `merging_scripts/`

Includes the scripts used to perform model merging across various techniques:

* **Linear Merging** (5 variants)
* **TIES-Merging**
* **Task Arithmetic (Addition)**
* **Model Breadcrumbs**
* **DARE + TIES Hybrid**

### 3. `SLAM-LLM/`

Adapted from the [SLAM-LLM repository](https://github.com/X-LANCE/SLAM-LLM), specifically utilizing the `asr_librispeech` example as a baseline framework. We have modified and tailored this implementation for our specific experimental pipeline.

### 4. `models/`

Stores the final ASR model checkpoints generated through our experiments:

* **Base Model**: The initial backbone model prior to merging.
* **Merged Models**: Checkpoints resulting from the merging methods, including designated Methods 1, 2, and 3.

---



## Usage

* **Model Merging:** Navigate to `merging_scripts/` to run the respective merging strategies on your checkpoints.
* **Intersection Correction:** Use the scripts under `InterCorrect_scripts/` to apply the correction phase.
* **ASR Evaluation/Training:** Follow the configuration instructions provided within the `SLAM-LLM/` directory for speech recognition tasks.

---

## Citation

If you use this code or our models in your research, please cite our preprint:

```bibtex
@misc{bravobravo2026intercorrectintersectionawarecorrectiondemographic,
      title={InterCorrect: Intersection-Aware Correction of Demographic Model Merging for Fair ASR}, 
      author={Ashley E. Bravo-Bravo and Yuchen Zhang and Haralambos Mouratidis and Ravi Shekhar and Monorama Swain},
      year={2026},
      eprint={2610.08604},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.08604}, 
}

## Acknowledgments

We acknowledge the authors of [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) for providing the foundational framework for our ASR experiments.
