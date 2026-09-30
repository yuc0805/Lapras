# Third-party notices

Lapras is released under the Apache License 2.0 (see `LICENSE`). It includes or adapts code from the
projects below; each file keeps its original copyright header where one exists.

| Component | Where | Upstream | License |
|---|---|---|---|
| LLaMA-Factory (training framework) | `lapras/` (data, model, hparams, train) | https://github.com/hiyouga/LLaMA-Factory | Apache-2.0 |
| ChatTS / ChatTS-Training | `lapras/backbones/chatts/`, ChatTS data plugin and template | https://github.com/NetManAIOps/ChatTS, https://github.com/xiezhe-24/ChatTS-Training | MIT (Qwen-derived modeling files: Apache-2.0) |
| OpenTSLM | `lapras/backbones/opentslm/` (port of OpenTSLM-Flamingo) | https://github.com/StanfordBDHG/OpenTSLM | MIT |
| OpenFlamingo | imported by the OpenTSLM-Flamingo backbone | https://github.com/mlfoundations/open_flamingo | MIT |
| SLIP | `lapras/backbones/slip/` | SLIP (Sensor Language-Informed Pretraining) | MIT |
| CODI | method this work builds on (self-distillation of continuous thoughts) | https://github.com/zhenyi4/codi | see upstream |
| COCONUT | baseline re-implemented in `lapras/train/coconut/` | https://github.com/facebookresearch/coconut | see upstream |
| iCoT (stepwise internalization) | baseline re-implemented in `lapras/train/icot/` | https://github.com/da03/Internalize_CoT_Step_by_Step | see upstream |
