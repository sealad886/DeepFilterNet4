# Dataset Attribution

This file records dataset sources and licenses used to build training data.
It is intentionally separate from the code license. Do not redistribute audio
unless the dataset license explicitly permits it.

## Summary (permissive-only)

| Dataset | License | Purpose | Notes |
| --- | --- | --- | --- |
| VCTK 0.92 | CC BY 4.0 | Clean speech | 48 kHz downsampled release. |
| LibriSpeech | CC BY 4.0 | Clean speech | Resample from 16 kHz to 48 kHz. |
| MUSAN | CC BY 4.0 | Noise + music | Use noise + music subsets. |
| FSD50K (filtered) | CC0 / CC BY | Outdoor/industrial/animals | Filter to CC0/CC-BY only. |
| AIR (Aachen IR) | MIT | RIR / speakerphone | Includes mobile phone RIRs. |
| OpenAIR | CC BY 4.0 | Room IRs | Real room impulse responses. |
| AcousticRooms (optional) | CC BY 4.0 | Synthetic RIR scale | Resample to 48 kHz. |

## Required attributions

Fill in the entries below before releasing weights or results.

### VCTK 0.92
- Source: https://datashare.ed.ac.uk/handle/10283/3443
- License: CC BY 4.0
- Citation/Attribution text:
  - Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019). CSTR VCTK Corpus:
    English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92), [sound].
    University of Edinburgh. The Centre for Speech Technology Research (CSTR).
    https://doi.org/10.7488/ds/2645

### LibriSpeech
- Source: https://us.openslr.org/12/
- License: CC BY 4.0
- Citation/Attribution text:
  - Panayotov, V., Chen, G., Povey, D., Khudanpur, S. (2015).
    LibriSpeech: an ASR corpus based on public domain audio books.
    IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

### MUSAN
- Source: https://www.openslr.org/17/
- License: CC BY 4.0
- Citation/Attribution text:
  - Snyder, D., Chen, G., Povey, D. (2015).
    MUSAN: A Music, Speech, and Noise Corpus. arXiv:1510.08484v1.

### FSD50K (filtered)
- Source: https://zenodo.org/records/4060432
- License: CC0 / CC BY (filtered)
- Filter rule: include only CC0/CC-BY clips, exclude CC-BY-NC and CC Sampling+
- Citation/Attribution text:
  - Fonseca, E., Favory, X., Pons, J., Font, F., Serra, X. (2022).
    FSD50K: An Open Dataset of Human-Labeled Sound Events.
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30, 829-852.

### AIR (Aachen Impulse Response Database)
- Source: https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
- License: MIT
- Citation/Attribution text:
  - Jeub, M., Schaefer, M., Vary, P. (2009).
    A Binaural Room Impulse Response Database for the Evaluation of Dereverberation Algorithms.
    International Conference on Digital Signal Processing (DSP).
  - Jeub, M., Schaefer, M., Krueger, H., Nelke, C. M., Beaugeant, C., Vary, P. (2010).
    Do We Need Dereverberation for Hand-Held Telephony?
    International Congress on Acoustics (ICA).

### OpenAIR
- Source: https://audeering.github.io/datasets/datasets/openair.html
- License: CC BY 4.0
- Citation/Attribution text:
  - OpenAIR dataset page (created by Joseph Rees-Jones and Damian Murphy, University of York).
    No formal citation provided on the dataset page; cite the dataset page and creators.

### AcousticRooms (optional)
- Source: https://github.com/facebookresearch/AcousticRooms
- License: CC BY 4.0
- Citation/Attribution text:
  - Liu, X., Kumar, A., Calamia, P., Amengual, S. V., Murdock, C., Ananthabhotla, I.,
    Robinson, P., Shlizerman, E., Ithapu, V. K., Gao, R. (2025).
    Hearing Anywhere in Any Environment. CVPR.

## Notes

- If you add any dataset beyond the list above, record it here.
- Keep a copy of each dataset's license text or a stable reference URL.
- If you publish weights, include this file (or a summarized version) alongside them.
