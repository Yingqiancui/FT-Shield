# FT-Shield
FT-Shield introduces a new method for providing copyright protection for digital images, preventing their unauthorized use in fine-tuning text-to-image diffusion models. This tool enables users to embed watermarks into images, ensuring that any unauthorized use can be detected and traced.

## Paper Reference
For detailed insights and the methodology behind FT-Shield, refer to the paper: [FT-Shield: A Watermark Against Unauthorized Fine-tuning in Text-to-Image Diffusion Models](https://arxiv.org/abs/2310.02401).

## Instruction
For generating watermarked images:

```python
python main.py --img_dir /path/to/your/img --captions_file /path/to/your/caption_files.jsonl
```

For training the watermark detector:

```python
python train_classifier.py --train_path /your/train/path
```
You can simply organize the training directory into two subfolders:
```
/your/train/path/
    ├── clean/
    └── watermarked/
```

For evaluating the watermark detector:
```python
python test.py --test_path /your/test/path
```
Similarly, you can simply organize the training directory into:
```
/your/test/path/
    ├── clean/
    └── watermarked/
```
