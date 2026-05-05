import os
os.environ['FLAGS_use_onednn'] = '0'  # 保险起见可以保留

from paddleocr import PaddleOCR
import pyautogui, time

ocr = PaddleOCR(lang='ch')
time.sleep(3)
screenshot = pyautogui.screenshot(region=(409,175,592,527))
screenshot.save('chat.png')
result = ocr.ocr('chat.png')
texts = [line[1][0] for line in result[0]] if result[0] else []
print(texts)
