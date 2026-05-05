import os
import time
import re
import pyautogui
import pyperclip
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from modelscope import AutoProcessor, AutoModelForImageTextToText
import warnings
warnings.filterwarnings("ignore")

# ======================== 配置区 ========================
# 1. 本地对话模型路径（Qwen2-0.5B-Instruct）
QWEN_MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"          # 改为你的实际路径

# 2. GLM-OCR 模型路径（已下载的文件夹）
GLM_OCR_PATH = "./ZhipuAI/GLM-OCR"                    # 改为你的实际路径

# 3. 微信聊天区域截图坐标 (左, 上, 宽, 高)
CHAT_REGION = (313, 89, 560, 405)
# 4. 微信输入框坐标
INPUT_BOX_X, INPUT_BOX_Y = 369, 537

# 5. 轮询间隔（秒），建议 ≥5
POLL_INTERVAL = 2

# 6. 对话模型回复参数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
SYSTEM_PROMPT = "你是KEFENG_STUDY_helper系统的AI,基于阿里云千问模型再此开发的程序"

# 7. 绿色气泡过滤参数（HSV 色彩空间）
GREEN_LOWER = (35, 40, 40)      # HSV 下界
GREEN_UPPER = (85, 255, 255)    # HSV 上界
GREEN_MASK_DILATE = 5           # 膨胀核大小
# =====================================================

# 设备配置
CHAT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCR_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"对话模型设备: {CHAT_DEVICE}， OCR 设备: {OCR_DEVICE}")

# ========== 加载对话模型 ==========
print("加载对话模型...")
chat_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
if chat_tokenizer.pad_token is None:
    chat_tokenizer.pad_token = chat_tokenizer.eos_token

if CHAT_DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    chat_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    chat_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
chat_model.eval()
print("对话模型就绪\n")

# ========== 加载 GLM-OCR ==========
print("加载 GLM-OCR 模型...")
ocr_processor = AutoProcessor.from_pretrained(GLM_OCR_PATH, trust_remote_code=True)
if OCR_DEVICE.startswith("cuda"):
    bnb_config_ocr = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        GLM_OCR_PATH,
        quantization_config=bnb_config_ocr,
        device_map=OCR_DEVICE,
        trust_remote_code=True,
    )
else:
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        GLM_OCR_PATH,
        device_map="cpu",
        trust_remote_code=True,
    )
ocr_model.eval()
print("GLM-OCR 就绪\n")

# ========== 工具函数 ==========
def remove_green_bubbles(image):
    """
    消除图片中的绿色气泡（自己的消息），返回处理后的 PIL 图片
    """
    img_np = np.array(image.convert("RGB"))
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    # 创建绿色掩码
    mask = cv2.inRange(img_hsv, GREEN_LOWER, GREEN_UPPER)
    # 膨胀掩码，覆盖气泡边缘
    kernel = np.ones((GREEN_MASK_DILATE, GREEN_MASK_DILATE), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # 将绿色区域填充为白色（背景色）
    img_np[mask > 0] = [255, 255, 255]
    return Image.fromarray(img_np)

def get_chat_texts(region):
    """
    截图、去绿色、AI 识别，返回识别到的文字列表
    """
    screenshot = pyautogui.screenshot(region=region)
    # 去除绿色气泡
    clean_img = remove_green_bubbles(screenshot)
    clean_img.save("chat_clean.png")  # 保存处理后的图片，方便检查

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": clean_img},
                {"type": "text", "text": "请识别图片中的所有聊天文字，按行输出，不要添加任何解释。"}
            ],
        }
    ]

    inputs = ocr_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(OCR_DEVICE) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = ocr_model.generate(**inputs, max_new_tokens=512)
    text = ocr_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 按行拆分，过滤无意义行
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # 额外过滤常见噪音
    noise = re.compile(r'^\d{1,2}:\d{2}$|对方正在输入|以上是打招呼内容')
    return [l for l in lines if not noise.match(l)]

def generate_reply(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = chat_tokenizer.encode(text, return_tensors="pt").to(chat_model.device)

    with torch.no_grad():
        outputs = chat_model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=chat_tokenizer.pad_token_id,
            eos_token_id=chat_tokenizer.eos_token_id,
        )
    return chat_tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()

def send_message(text):
    pyautogui.click(INPUT_BOX_X, INPUT_BOX_Y)
    time.sleep(0.2)
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('delete')
    pyperclip.copy(text)
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.2)
    pyautogui.press('enter')

# ========== 主循环 ==========
prev_texts = set()
last_replied = ""

print(f"开始监控微信新消息，每 {POLL_INTERVAL} 秒检查一次。（按 Ctrl+C 停止）")
try:
    while True:
        current_texts = get_chat_texts(CHAT_REGION)
        new_texts = [t for t in current_texts if t not in prev_texts]

        if new_texts:
            question = new_texts[-1]  # 取最后一条新消息
            if question == last_replied:
                print(f"\n[重复消息] {question}，跳过")
            else:
                print(f"\n检测到：{question}")
                reply = generate_reply(question)
                print(f"回复：{reply}")
                send_message(reply)
                last_replied = question
            # 更新已见文字集
            time.sleep(0.5)
            prev_texts = set(get_chat_texts(CHAT_REGION))
        else:
            prev_texts = set(current_texts)
            print('.', end='', flush=True)

        time.sleep(POLL_INTERVAL)
except KeyboardInterrupt:
    print("\n已停止。")
