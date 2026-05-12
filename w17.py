# Required: pip install torch torchvision torchaudio "numpy<2.0" "transformers>=4.40.0" accelerate bitsandbytes facenet-pytorch opencv-python pillow streamlit googlesearch-python groq fpdf2

import re
import os
import cv2
import json
import torch
import tempfile
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from fpdf import FPDF
from googlesearch import search
from facenet_pytorch import MTCNN
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification

# =====================================
# 1. CONFIG & SYSTEM SETUP
# =====================================
st.set_page_config(page_title="Sentinel OSINT v7.0", page_icon="🛡️", layout="wide")

DEVICE = 0 if torch.cuda.is_available() else -1
TARGET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL_CONTEXT = "facebook/bart-large-mnli" 
TEXT_MODEL_STRUCTURAL = "mrm8488/bert-tiny-finetuned-fake-news-detection" 
IMAGE_DETECTOR_MODEL = "umm-maybe/AI-image-detector" 
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"

# =====================================
# 2. ADVANCED STYLING
# =====================================
st.markdown("""
<style>
    .main-header {font-size:40px; font-weight:800; color:#00ffb3; margin-bottom:0px;}
    .sub-header {font-size:16px; color:#888; margin-bottom:20px;}
    .result-card {padding:20px; border-radius:10px; background:#111827; border:1px solid #333; color:#ffffff; margin-bottom:15px;}
    .custom-model-card {padding:20px; border-radius:10px; background:#1e1e2f; border:2px solid #00ffb3; color:#ffffff; margin-bottom:20px; margin-top: 10px;}
    .label-real {color:#00ff99; font-size:20px; font-weight:bold;}
    .label-fake {color:#ff4b4b; font-size:20px; font-weight:bold;} 
    .image-caption-box {background-color: #1a202c; padding: 12px; border-radius: 0px 0px 8px 8px; border-top: 2px solid #4a5568; margin-bottom: 20px; color: #f8fafc; font-size: 15px;}
    .groq-box {background-color: #1a202c; padding: 20px; border-radius: 8px; border: 2px solid #f59e0b; margin-top: 20px; margin-bottom: 20px; color: #f8fafc;}
    .metric-badge {padding: 8px 12px; border-radius: 5px; margin-bottom: 5px; font-size: 14px; font-weight: bold;}
    .badge-pass {background-color: #064e3b; color: #34d399; border: 1px solid #059669;}
    .badge-fail {background-color: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626;}
    .badge-warn {background-color: #78350f; color: #fbbf24; border: 1px solid #d97706;}
</style>
""", unsafe_allow_html=True)

# =====================================
# 3. UTILITY & FORENSIC FUNCTIONS
# =====================================
def anti_leakage_scrubber(text):
    text = str(text)
    text = text.replace("TITLE: ", "")
    text = re.sub(r'=+', '', text)
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(AP\)', '', text)
    text = re.sub(r'^[A-Z\s/]+ \-\s*', '', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_for_deepfake(image, mtcnn_model):
    try:
        img_draw = image.convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR)
        boxes, _ = mtcnn_model.detect(img_draw)
        
        original_with_box_pil = image.convert("RGB") 
        processed_face_pil = None
        face_detected = False

        if boxes is not None and len(boxes) > 0:
            largest_box = None
            max_area = 0
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box = box
            
            if max_area < 4000: 
                return original_with_box_pil, processed_face_pil, face_detected
                
            face_detected = True
            box = largest_box.astype(int)
            h, w = img_cv.shape[:2]
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            
            draw = ImageDraw.Draw(original_with_box_pil)
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 130, 255), width=5)

            face_cv = img_cv[y1:y2, x1:x2]
            if face_cv.size > 0:
                face_cv = cv2.resize(face_cv, (224, 224))
                processed_face_pil = Image.fromarray(cv2.cvtColor(face_cv, cv2.COLOR_BGR2RGB))
            else:
                processed_face_pil = None
                
        return original_with_box_pil, processed_face_pil, face_detected
    except Exception as e:
        return image, None, False

def pdf_sanitize(text):
    text = str(text).replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2022', '-').replace('\n\n', '\n')
    text = text.replace('<br>', '\n').replace('**', '') 
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(news_text, verdict, reasons, groq_report, visual_results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(w=0, h=10, text="Sentinel OSINT - Intelligence Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(w=0, h=10, text=f"System Verdict: {verdict}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", '', 11)
    
    if reasons:
        for r in reasons:
            clean_r = pdf_sanitize(r.replace('*', ''))
            pdf.write(txt=f"{clean_r}\n")
            pdf.ln(2)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(w=0, h=10, text="Llama-3 Logical Analysis:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", '', 11)
    pdf.write(txt=pdf_sanitize(groq_report) + "\n\n")
    pdf.ln(5)

    if visual_results:
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(w=0, h=10, text="Visual Forensics Log:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", '', 11)
        for idx, res in enumerate(visual_results):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img_for_pdf = res['original_with_box_pil'].convert("RGB")
                img_for_pdf.thumbnail((500, 500)) 
                img_for_pdf.save(tmp.name)
                
                pdf.image(tmp.name, w=90)
                pdf.ln(2)
                cap = pdf_sanitize(f"Caption: {res['caption']}")
                pdf.write(txt=cap + "\n")
                if res.get('illogical_reason'):
                    ill = pdf_sanitize(f"Logic Failure: {res['illogical_reason']}")
                    pdf.write(txt=ill + "\n")
                pdf.ln(10)
    
    return bytes(pdf.output())

# =====================================
# 4. ENGINE INITIALIZATION
# =====================================
@st.cache_resource
def initialize_sentinel():
    t_engine_ctx = pipeline("zero-shot-classification", model=TEXT_MODEL_CONTEXT, device=DEVICE)
    t_engine_str = pipeline("text-classification", model=TEXT_MODEL_STRUCTURAL, device=DEVICE)
    v_detector = pipeline("image-classification", model=IMAGE_DETECTOR_MODEL, device=DEVICE)
    b_proc = BlipProcessor.from_pretrained(CAPTION_MODEL)
    b_mod = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(TARGET_DEVICE)
    mtcnn_model = MTCNN(keep_all=False, device=TARGET_DEVICE)
    
    custom_tokenizer = AutoTokenizer.from_pretrained(".")
    custom_model = AutoModelForSequenceClassification.from_pretrained(".")
    custom_model.to(TARGET_DEVICE)
    custom_model.eval()
    
    return t_engine_ctx, t_engine_str, v_detector, b_proc, b_mod, mtcnn_model, custom_tokenizer, custom_model

with st.spinner("Synchronizing Neural Pipelines & Face Extractors..."):
    text_ctx, text_str, img_detector, blip_p, blip_m, mtcnn_model, custom_tokenizer, custom_model = initialize_sentinel()


# =====================================
# 5. UI INTERFACE
# =====================================
st.markdown('<div class="main-header">🛡️ Sentinel OSINT Command (v7.0)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Custom DistilBERT • External Verification • Groq Reality Logic Check</div>', unsafe_allow_html=True)

unified_uploads = st.file_uploader("Upload mixed batch", type=["txt", "jpg", "jpeg", "png", "webp"], accept_multiple_files=True, label_visibility="collapsed")

extracted_text = ""
extracted_images = []

if unified_uploads:
    for f in unified_uploads:
        if f.name.endswith('.txt'):
            extracted_text = f.read().decode('utf-8', errors='ignore')
        else:
            extracted_images.append(f)

col_a, col_b = st.columns(2)
with col_a:
    text_input = st.text_area("Narrative Text", value=extracted_text, height=200, placeholder="Text will appear here...")
with col_b:
    manual_imgs = st.file_uploader("Add/Override Images (Optional)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    final_images = extracted_images.copy()
    if manual_imgs: final_images.extend(manual_imgs)
    
    if final_images: st.success(f"✅ {len(final_images)} image(s) successfully loaded into forensic pipeline.")

if st.button("🏁 EXECUTE PIPELINE ANALYSIS", use_container_width=True):
    if not text_input and not final_images:
        st.error("Protocol Error: Please provide either Text or Image(s) to analyze.")
    else:
        st.divider()
        
        has_text = bool(text_input)
        has_images = bool(final_images)
        clean_text = anti_leakage_scrubber(text_input) if has_text else ""
        
        is_structural_factual, is_context_factual = True, True
        str_factual_score = 0.0
        ctx_top_label = "None"
        visual_analysis_results = []
        any_visual_fakes = False
        custom_is_fake = False
        custom_conf_fake = 0.0
        custom_conf_true = 0.0
        
        groq_report = "No logical analysis performed."
        groq_visual_fails_reality = False
        groq_news_fake = False

        # --- PHASE 1: CUSTOM DISTILBERT MODEL ---
        if has_text:
            with st.spinner("Evaluating via Custom DistilBERT Model..."):
                inputs = custom_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
                if "token_type_ids" in inputs: del inputs["token_type_ids"]
                inputs = {k: v.to(TARGET_DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = custom_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]
                custom_conf_fake = probabilities[1].item() * 100 
                custom_conf_true = probabilities[0].item() * 100
                custom_is_fake = (torch.argmax(outputs.logits, dim=1).item() == 1)

            # --- PHASE 2: EXTERNAL OSINT ENGINES ---
            with st.spinner("Running External Multi-Model Logic..."):
                disinfo_labels = ["objective factual reporting", "sensationalist clickbait", "pseudoscience and miracle cures", "fabricated political hoax", "absurd, illogical, or physically impossible event"]
                ctx_res = text_ctx(clean_text, disinfo_labels, truncation=True)
                ctx_top_label = ctx_res['labels'][0]
                str_res = text_str(clean_text, truncation=True, max_length=512)
                str_factual_score = str_res[0]['score'] if ("0" in str_res[0]['label'] or "REAL" in str_res[0]['label'].upper()) else (1.0 - str_res[0]['score'])
                
                is_structural_factual = str_factual_score >= 0.50
                is_context_factual = ctx_top_label not in ["fabricated political hoax", "absurd, illogical, or physically impossible event", "sensationalist clickbait", "pseudoscience and miracle cures"]

        # --- PHASE 3: VISUAL INTEGRITY ---
        if has_images:
            with st.spinner(f"Running Forensics on {len(final_images)} image(s)..."):
                for img_file in final_images:
                    try:
                        original_img = Image.open(img_file).convert("RGB")
                        original_with_box_pil, processed_face_pil, face_detected = preprocess_for_deepfake(original_img, mtcnn_model)
                        
                        v_res = img_detector(original_img)
                        v_label = v_res[0]['label'].upper()
                        if "FAKE" in v_label or "ARTIFICIAL" in v_label:
                            v_label = "FAKE"
                        else:
                            v_label = "REAL"
                            
                        v_confidence = v_res[0]['score']
                        
                        c_in = blip_p(original_img, return_tensors="pt").to(blip_m.device)
                        c_out = blip_m.generate(**c_in, max_length=40)
                        img_caption = blip_p.decode(c_out[0], skip_special_tokens=True)
                        
                        # Removed the strict > 0.80 threshold. If the Swin-Transformer leans FAKE, we flag it.
                        is_visual_fake = False
                        if v_label == "FAKE": 
                            any_visual_fakes = True
                            is_visual_fake = True
                        
                        visual_analysis_results.append({
                            "filename": img_file.name, 
                            "original_img": original_img, 
                            "processed_face": processed_face_pil, 
                            "original_with_box_pil": original_with_box_pil,
                            "label": "FAKE" if is_visual_fake else "REAL", 
                            "confidence": v_confidence,
                            "caption": img_caption, 
                            "human_found": face_detected, 
                            "illogical_reason": ""
                        })
                    except Exception as e:
                        print(f"Error processing visual image {img_file.name}: {e}")
                        pass

        # --- PHASE 4: GROQ LLAMA-3 REALITY & LOGIC CHECK ---
        with st.spinner("Consulting Groq Llama-3 Fact Checker..."):
            try:
                with open("AI_model_api.json", "r") as f:
                    groq_key = json.load(f).get("GROQ_API_KEY", "")
                
                if groq_key and (has_text or has_images):
                    from groq import Groq
                    client = Groq(api_key=groq_key)
                    
                    prompt = "You are an elite OSINT Intelligence Analyst and Fact Checker. Use your internal knowledge of the real world to verify the following:\n\n"
                    if has_text: prompt += f"News Narrative: '{clean_text}'\n"
                    if has_images:
                        caps = [res['caption'] for res in visual_analysis_results]
                        prompt += f"Visual Evidence (Image Captions): {caps}\n"
                    
                    prompt += "\nInstructions:\n"
                    if has_text:
                        prompt += "1. News Check: Does this text describe a fake, satirical, impossible, or scam narrative (like free energy generators, replacing cops with cats, etc)? Set 'news_is_fake' to true if it is not a credible real-world event.\n"
                    if has_images:
                        prompt += "2. Visual Logic Check: Does the caption describe an AI-generated trope (e.g. perfect glowing lab equipment, impossible physics, animals in human roles) or contradict the real world? Do NOT excuse it as a joke. If it depicts an impossible reality for a serious news event, you MUST set 'visual_fails_reality' to true.\n"
                    
                    prompt += """
                    Respond ONLY in this exact JSON format:
                    {
                    """
                    if has_text: prompt += '  "news_analysis": "Your strict reality check of the text.",\n  "news_is_fake": true/false'
                    if has_text and has_images: prompt += ",\n"
                    if has_images: prompt += '  "visual_analysis": "Your strict reality check of the visuals.",\n  "visual_fails_reality": true/false\n'
                    prompt += "}"
                    
                    comp = client.chat.completions.create(
                        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}],
                        temperature=0.1, response_format={"type": "json_object"}
                    )
                    
                    res_data = json.loads(comp.choices[0].message.content)
                    
                    groq_news_analysis = res_data.get("news_analysis", "No text provided.")
                    groq_news_fake = res_data.get("news_is_fake", False)
                    groq_visual_analysis = res_data.get("visual_analysis", "No images provided.")
                    groq_visual_fails_reality = res_data.get("visual_fails_reality", False)
                    
                    groq_report = ""
                    if has_text:
                        groq_report += f"**News Reality Check:** {groq_news_analysis}<br><br>"
                    if has_images:
                        groq_report += f"**Visual Logic Check:** {groq_visual_analysis}"
                    
                    if (groq_visual_fails_reality or groq_news_fake) and has_images:
                        any_visual_fakes = True
                        for vr in visual_analysis_results:
                            if not vr['illogical_reason']:
                                if groq_visual_fails_reality:
                                    vr['illogical_reason'] = groq_visual_analysis
                                else:
                                    vr['illogical_reason'] = "Contextual Failure: The foundational news is false, impossible, or satire. Visuals are therefore invalid and fabricated."
            except Exception as e:
                groq_report = f"Groq API skipped or failed: {str(e)}"

        # =====================================
        # DASHBOARD RENDERING
        # =====================================
        st.subheader("📊 Intelligence Report Dashboard")
        
        # 1. TOP TIER: CUSTOM DISTILBERT (WELFake)
        if has_text:
            st.markdown("### 🎯 Custom DistilBERT Engine (WELFake Dataset)")
            if custom_is_fake:
                st.markdown(f'''<div class="custom-model-card" style="border-color: #ff4b4b;">
                    <div style="color:#ff4b4b; font-size:18px; font-weight:bold;">🚨 DISTILBERT MODEL: FAKE NEWS DETECTED</div>
                    <p style="margin-top: 5px;"><b>Fake Probability:</b> {custom_conf_fake:.2f}% | <b>True Probability:</b> {custom_conf_true:.2f}%</p>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="custom-model-card" style="border-color: #00ff99;">
                    <div style="color:#00ff99; font-size:18px; font-weight:bold;">✅ DISTILBERT MODEL: TRUE NEWS VERIFIED</div>
                    <p style="margin-top: 5px;"><b>True Probability:</b> {custom_conf_true:.2f}% | <b>Fake Probability:</b> {custom_conf_fake:.2f}%</p>
                </div>''', unsafe_allow_html=True)

        # 2. MIDDLE TIER: EXTERNAL ENGINES
        st.markdown("### 🌐 External OSINT Engines")
        metrics_cols = st.columns(3)
        if has_text:
            metrics_cols[0].metric("External Integrity Score", f"{str_factual_score*100:.1f}%", "Pass" if is_structural_factual else "- Fail")
            metrics_cols[1].metric("Contextual Flavor", ctx_top_label.title(), "Pass" if is_context_factual else "- Fail")
        
        if has_images:
            vis_label = "ILLOGICAL / SYNTHETIC" if any_visual_fakes else "VERIFIED"
            vis_status = "- Fail" if any_visual_fakes else "Pass"
            metrics_cols[2].metric("Visual/Logical Authenticity", vis_label, vis_status)
        else:
            metrics_cols[2].metric("Visual/Logical Authenticity", "N/A", "Off")

        # 3. BOTTOM TIER: GROQ FACT CHECK
        st.markdown(f'<div class="groq-box"><b>🧠 Advanced Fact Checking & Logic (Llama-3.1):</b><br><br>{groq_report}</div>', unsafe_allow_html=True)

        # 4. OVERALL VERDICT LOGIC
        st.markdown("### 🛑 Overall System Verdict")
        is_manipulated = False
        reasons = []

        if has_text and custom_is_fake:
            is_manipulated = True
            reasons.append("• **DistilBERT Engine:** Detected linguistic patterns consistent with Fake News.")
            
        if has_text and not is_structural_factual:
            is_manipulated = True
            reasons.append(f"• **External Structural Engine:** Text lacks factual integrity (Score: {str_factual_score*100:.1f}%).")
            
        if has_text and not is_context_factual:
            is_manipulated = True
            reasons.append(f"• **External Contextual Engine:** Flagged text as '{ctx_top_label.title()}'.")

        if groq_news_fake:
            is_manipulated = True
            reasons.append("• **Llama-3 Fact Checker:** The news narrative is factually false or absurd in the real world.")

        if groq_visual_fails_reality:
            is_manipulated = True
            reasons.append("• **Llama-3 Logic Engine:** The visual evidence is absurd and fails the reality check.")

        if any_visual_fakes and not groq_visual_fails_reality and not groq_news_fake:
            is_manipulated = True
            reasons.append("• **Visual Forensics:** Images detected as synthetic, contextually impossible, or heavily manipulated.")

        if has_text and is_structural_factual and is_context_factual and not groq_visual_fails_reality and not groq_news_fake and not any_visual_fakes:
            is_manipulated = False
            reasons = []

        verdict_text = "MANIPULATION DETECTED" if is_manipulated else "CONTENT VERIFIED"
        
        if is_manipulated:
            st.markdown(f'''<div class="result-card" style="border-left: 4px solid #ff4b4b;">
                <div class="label-fake">⚠️ MANIPULATION DETECTED</div>
                <p style="margin-top:10px;">Pipeline flagged this content:<br><br>{"<br>".join(reasons)}</p>
            </div>''', unsafe_allow_html=True)
        else:
             st.markdown(f'''<div class="result-card" style="border-left: 4px solid #00ff99;">
                <div class="label-real">✅ CONTENT VERIFIED</div>
                <p style="margin-top:10px;">The content is valid. High-confidence reporting from the OSINT framework verified the narrative.</p>
            </div>''', unsafe_allow_html=True)

        # PDF DOWNLOAD BUTTON
        pdf_bytes = create_pdf_report(clean_text, verdict_text, reasons, groq_report, visual_analysis_results)
        st.download_button(label="📄 Download Intelligence Report (PDF)", data=pdf_bytes, file_name="Sentinel_Report.pdf", mime="application/pdf", use_container_width=True)

        # VISUAL EVIDENCE LOG
        if has_images:
            st.markdown("---")
            st.markdown("### 🖼️ Visual Forensics & Logic")
            for result in visual_analysis_results:
                st.markdown(f"**Analyzing File:** `{result['filename']}`")
                
                # --- NEW INDEPENDENT DUAL-UI REPORTING ---
                
                # 1. Groq Contextual Analysis Badge
                if result['illogical_reason']:
                    st.markdown(f'<div class="metric-badge badge-fail">🚨 Llama-3 Context Scanner: FAILED - {result["illogical_reason"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-badge badge-pass">✅ Llama-3 Context Scanner: PASSED - No logical anomalies detected</div>', unsafe_allow_html=True)
                
                # 2. Deepfake Pixel Analysis Badge
                if not result['human_found']:
                    st.markdown('<div class="metric-badge badge-warn">🔎 Deepfake Pixel Scanner: SKIPPED - No clear human face detected</div>', unsafe_allow_html=True)
                else:
                    if result['label'] == "FAKE":
                        st.markdown(f'<div class="metric-badge badge-fail">🚨 Deepfake Pixel Scanner: FAILED - AI Manipulation Detected ({result["confidence"]*100:.1f}% confidence)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-badge badge-pass">✅ Deepfake Pixel Scanner: PASSED - Appears physically authentic ({result["confidence"]*100:.1f}% confidence)</div>', unsafe_allow_html=True)

                # 3. Overall Verdict Badge (If either fails, the image is rejected)
                if result['illogical_reason'] or result['label'] == "FAKE":
                    st.error("❌ **OVERALL VISUAL VERDICT: REJECTED (MANIPULATION DETECTED)**")
                else:
                    st.success("✅ **OVERALL VISUAL VERDICT: VERIFIED (AUTHENTIC)**")

                # Image Display
                c1, c2 = st.columns(2)
                with c1:
                    st.image(result['original_with_box_pil'], caption="Target Identification", use_container_width=True)
                with c2:
                    if result['processed_face'] is not None:
                        st.image(result['processed_face'], caption="Forensic Face Scan", use_container_width=True)
                
                st.markdown(f'<div class="image-caption-box">🤖 <b>Detected Scene:</b> {result["caption"].capitalize()}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Sentinel OSINT Pipeline v7.0")