# VisouraReID
VisouraReID: Modular Supervision for Person Re-Identification

## 🚩 News & Timeline
- **[2026-08-09]**: 🚀 Released the initial source code and published dataset / pretrained model checkpoints on [Hugging Face](https://huggingface.co/David-Magdy/VisouraReID).
- **[Pending]**: ✍️ Paper writing in progress.
- **[Pending]**: 📄 Paper publication and preprint release.

## 📊 Progress & Roadmap

| Milestone / Feature | Status | Date | Note |
| :--- | :---: | :---: | :--- |
| **Source Code Publishing** | ✅ Completed | 2026-08-09 | Initial release of codebase and fine-tuning/inference pipeline |
| **Data & Model Weights Publishing** | ✅ Completed | 2026-08-09 | Checkpoints and training logs released on [Hugging Face](https://huggingface.co/David-Magdy/VisouraReID) |
| **Paper Writing** | ⏳ In Progress | Pending | Manuscript preparation |
| **Paper Publishing** | ⏳ Pending | Pending | ArXiv preprint & publication |

## Contributions
## Results
![VisouraReID](pics/sota_pic.png)
![VisouraReID-tb](pics/sota_table.png)
## Download
You can download person ReID supervised-trained model and log from [reid_ft_model_logs](https://huggingface.co/David-Magdy/VisouraReID)

## ReID Fine-tuning and  Evaluating
first download the pretrained models from [ViT-S/16](https://huggingface.co/lakeAGI/PersonViT/tree/main/vits.lup.256x128.wopt.csk.4-8.ar.375.n8) and save it to pretrained
```shell
cd transreid_pytorch
sh run_epochs.sh ../pretrained/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/ vits.lup.256x128.wopt.csk.4-8.ar.375.n8 220 0 2 small
```
