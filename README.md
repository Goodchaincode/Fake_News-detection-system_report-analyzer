# Fake_News-detection-system_report-analyzer
This system will help us to detect any news fake or true , which are posted online.

Its trained model could not be uploaded on the github due to size issue 
Put API key in .json file , and you can integrate your own trained model also inplace of my distilBert trained mode .

# Sentinel OSINT Pipeline v7.0: Fake News & Disinformation Analyzer

**Sentinel OSINT Pipeline v7.0** is a sophisticated Open Source Intelligence (OSINT) tool designed to detect disinformation and fake news through a multi-layered verification process. It goes beyond standard text analysis by performing a comprehensive "visual autopsy" and logical cross-referencing to evaluate both textual narratives and visual content.

### Key Features
* **Multi-Modal Analysis:** Evaluates text and images simultaneously to identify factual inconsistencies.
* **Custom DistilBERT Engine:** Utilizes a highly optimized, locally trained DistilBERT model (99.53% accuracy on the WELFake dataset) designed for OOM-Proof execution.
* **Zero-Shot Disinformation Categorization:** Employs `bart-large-mnli` to classify text into specific disinformation categories (e.g., "sensationalist clickbait", "political hoax").
* **Visual Forensics Suite:**
  * Detects synthetic/AI-generated images using a specialized AI Image Detector.
  * Performs forensic face scanning and extraction using MTCNN.
  * Generates natural language image descriptions via BLIP for logical verification.
* **LLM Fact-Checking:** Integrates Llama-3.1 (via Groq API) for a final "reality check" against real-world knowledge.
* **Dynamic Reporting:** Automatically generates comprehensive PDF reports of the forensic findings.

---

### Installation Guide

To run this project locally, ensure you have Python installed, and then install the required dependencies. It is highly recommended to set up a virtual environment first.

Run the following commands in your terminal:

```bash
# 1. Core Machine Learning and Computer Vision Models
pip install torch torchvision torchaudio "numpy<2.0" "transformers>=4.40.0" accelerate bitsandbytes facenet-pytorch opencv-python pillow

# 2. Data Analysis and Visualization Tools
pip install scikit-learn pandas umap-learn wordcloud

# 3. Web Framework, APIs, and Reporting
pip install streamlit googlesearch-python groq fpdf2

