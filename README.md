# DirectAudioEdit: Inversion-Free Text-Guided Audio Editing via Diffusion Prediction Contrast


[![arXiv](https://img.shields.io/badge/arXiv-2606.07356-B31B1B.svg)](https://arxiv.org/abs/2606.07356)
[![Demo](https://img.shields.io/badge/Demo-Page-4CAF50.svg)](https://niutrans.github.io/DirectAudioEdit)



This repository contains the official code release for ***DirectAudioEdit: Inversion-Free Text-Guided Audio Editing via Diffusion Prediction Contrast***.

## Abstract
Text-guided audio editing aims to modify the language-specified acoustic content while preserving edit-irrelevant source components. Existing training-free methods typically rely on inversion-based editing. While inversion-free editing is appealing as it decreases computational overhead and reconstruction errors, it remains largely unexplored for audio editing. The key challenge is to construct a source-to-target editing trajectory through diffusion denoising dynamics. In this paper, we introduce ***DirectAudioEdit***, the first attempt to develop a training-free and inversion-free method for audio editing.

## Demo Page

Visit [**niutrans.github.io/DirectAudioEdit**](https://niutrans.github.io/DirectAudioEdit) to listen to audio editing samples and compare DirectAudioEdit with baseline methods (DDIM-inv, DDPM-inv, SDEdit).


## Requirements

```bash
python -m pip install -r requirements.txt
```

## Usage Example

```bash
CUDA_VISIBLE_DEVICES=0 python ./code/main_run.py --init_aud example/a_baby_is_crying.wav --source_prompt "a baby is crying" --target_prompt "A cat is meowing" --results_path ./results
```


## Citation

```bibtex
@article{directaudioedit2026,
  title={DirectAudioEdit: Inversion-Free Text-Guided Audio Editing via Diffusion Prediction Contrast},
  author={Ge, Zhengkun and Liu, Xiaoqian and Zhang, Haoran and Ge, Yuan and Zhang, Junxiang and Yu, Zhengtao and Zhu, Jingbo and Xiao, Tong},
  journal={arXiv preprint arXiv:2606.07356},
  year={2026}
}

```



## Acknowledgements

Parts of this code are heavily based on [AudioEditingCode](https://github.com/HilaManor/AudioEditingCode)

AudioLDM2 is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa]. Therefore, using the weights of AudioLDM2 (the default) and code originating in the `code/audioldm` folder is under the same license (eg., `utils.py:load_audio` uses code from `code/audioldm`).  
The rest of the code (inversion, PCs computation) is licensed under an MIT license.


[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
