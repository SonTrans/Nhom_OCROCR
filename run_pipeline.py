import os
import pytesseract
from PIL import Image

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

IMAGE_DIR = "test_images"
OUTPUT_DIR = "ocr_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
vietocr_engine = Predictor(config)

images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
output_file_path = os.path.join(OUTPUT_DIR, "summary_report.txt")

with open(output_file_path, "w", encoding="utf-8") as f:
    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        print(f"-> Processing: {img_name}")

        f.write(f"=== IMAGE: {img_name} ===\n")
        pil_img = Image.open(img_path)

        #Tesseract
        f.write("1. Tesseract\n")
        try:
            tess_raw = pytesseract.image_to_string(pil_img, lang="vie+eng")
            f.write(f"{tess_raw.strip()}\n")
        except Exception as e:
            f.write(f"Loi: {e}\n")

        f.write("\n")

        # VietOCR
        f.write("2. VietOCR\n")
        try:
            viet_raw = vietocr_engine.predict(pil_img)
            f.write(f"{str(viet_raw).strip()}\n")
        except Exception as e:
            f.write(f"Loi: {e}\n")

        f.write("\n")

        #PaddleOCR
        f.write("3. PaddleOCR\n")
        try:
            paddle_engine = PaddleOCR(use_textline_orientation=True, lang='en')
            paddle_raw = paddle_engine.ocr(img_path, cls=True)
            f.write(f"{str(paddle_raw).strip()}\n")
        except Exception as e:
            err_msg = str(e).splitlines()[0] if e else "Unknown Error"
            f.write(f"Loi: {err_msg}\n")

        f.write("\n" + "=" * 40 + "\n\n")

print(f"🎉 Done: {output_file_path}")