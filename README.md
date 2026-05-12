# Fake_News-detection-system_report-analyzer

This system helps detect whether online news content is true or fake.

## DistilBERT Model Training Summary

- **Training objective:** Fake news classification on the **WELFake Dataset** (64,912 articles)
- **Architecture:** `distilbert-base-uncased`
- **Performance:** Final test accuracy of **99.53%** after **3 epochs**
- **Optimization:** Training pipeline designed to be **OOM-proof** for dual-GPU execution

## Pretrained Models Used in the Project

### 1) Text Analysis Models

- **BART-Large-MNLI:** Used for zero-shot classification to evaluate text against disinformation labels (for example, sensationalist clickbait and fabricated political hoax) without category-specific retraining.
- **BERT-Tiny (fine-tuned):** Compact BERT variant fine-tuned for fake news detection to perform structural text-integrity analysis.
- **DistilBERT (custom-loaded):** Primary narrative analysis engine aligned with the WELFake dataset, providing "True" vs "Fake" probability scores.
- **Llama-3.1-8B-Instant (via Groq API):** Advanced fact-checking model that compares news narratives and image captions against world knowledge to identify logical inconsistencies.

### 2) Visual and Forensic Models

- **AI Image Detector:** Image classifier for distinguishing authentic photos from synthetic (AI-generated) images.
- **BLIP (Bootstrapping Language-Image Pre-training):** Image captioning model used to generate descriptions that are passed to the Llama-3 logic step for plausibility checks.
- **MTCNN (Multi-task Cascaded Convolutional Networks):** Face-detection framework used for forensic face extraction (detect, box, crop) before manipulation analysis.
