from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss  # ใช้ FAISS สำหรับการค้นหาความคล้ายคลึง
import json
import requests
from pyngrok import ngrok  # สำหรับการเชื่อมต่อ ngrok

# โหลดโมเดล sentence transformer
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# รายละเอียดการเชื่อมต่อ Neo4j
URI = "neo4j://localhost"
AUTH = ("neo4j", "0986576621")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()

cypher_query = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''

# ดึงข้อความทักทายจากฐานข้อมูล Neo4j
greeting_corpus = []
results = run_query(cypher_query)
for record in results:
    greeting_corpus.append(record['name'])

# ทำให้ข้อมูลทักทายไม่ซ้ำกัน
greeting_corpus = list(set(greeting_corpus))
print(greeting_corpus)

# แปลงข้อความทักทายเป็นเวกเตอร์ด้วยโมเดล sentence transformer
greeting_vecs = model.encode(greeting_corpus, convert_to_numpy=True, normalize_embeddings=True)

# สร้าง FAISS index
d = greeting_vecs.shape[1]  # ขนาดของเวกเตอร์
index = faiss.IndexFlatL2(d)  # ใช้ L2 distance สำหรับการค้นหา (คล้ายกับ cosine similarity)
index.add(greeting_vecs)  # เพิ่มเวกเตอร์ลงใน FAISS index

def compute_similar_faiss(sentence):
    # แปลงข้อความที่ต้องการค้นหาเป็นเวกเตอร์
    ask_vec = model.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)
    # ค้นหาข้อความที่คล้ายที่สุดใน FAISS index
    D, I = index.search(ask_vec, 1)  # คืนค่าที่คล้ายที่สุด 1 อันดับ
    return D[0][0], I[0][0]

def neo4j_search(neo_query):
    results = run_query(neo_query)
    if results:
        return results[0]['answer']  # เปลี่ยนเป็น 'answer' สำหรับ TrafficRule
    return "ไม่พบข้อมูลเกี่ยวกับกฎจราจรที่ถามมา"

# Ollama API endpoint (สมมติว่าคุณกำลังใช้ Ollama ในเครื่อง)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json"
}

def llama_generate_response(prompt):
    # เตรียมข้อมูลสำหรับคำขอไปยังโมเดล supachai/llama-3-typhoon-v1.5
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",  
        "prompt": prompt,
        "stream": False
    }

    # ส่งคำขอ POST ไปยัง Ollama API
    response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))

    # ตรวจสอบว่าคำขอสำเร็จหรือไม่
    if response.status_code == 200:
        # แปลงข้อมูล JSON ที่ได้รับ
        response_data = response.text
        data = json.loads(response_data)
        decoded_text = data.get("response", "ไม่พบข้อความตอบกลับ")
        return decoded_text
    else:
        # จัดการข้อผิดพลาด
        print(f"ไม่สามารถรับข้อความตอบกลับได้: {response.status_code}, {response.text}")
        return "เกิดข้อผิดพลาดระหว่างการสร้างข้อความตอบกลับ"

def compute_response(sentence):
    score, index = compute_similar_faiss(sentence)
    print(f"Score: {score}, Index: {index}")
    
    if score < 0.5:  # กรณีข้อความเป็นการทักทาย
        Match_greeting = greeting_corpus[index]
        My_cypher = f"MATCH (n:Greeting) WHERE n.name = '{Match_greeting}' RETURN n.msg_reply as reply"
        my_msg = run_query(My_cypher)  # ดึงข้อความทักทายจาก Neo4j
        if my_msg:
            return my_msg[0]['reply']
        else:
            return "ไม่พบข้อความทักทายที่ตรงกัน"
    else:
        # ค้นหาจาก Neo4j สำหรับกฎจราจร
        My_cypher = f"MATCH (q:TrafficRule) WHERE q.question CONTAINS '{sentence}' RETURN q.answer as answer"
        my_msg = run_query(My_cypher)  # ดึงข้อมูลจาก Neo4j
        if my_msg:
            return my_msg[0]['answer']  # ส่งคืนข้อมูลจาก Neo4j
        else:
            # กรณีไม่พบคำตอบใน Neo4j, เรียกใช้ Ollama เพื่อสร้างคำตอบ
            print("ไม่พบข้อมูลใน Neo4j, เรียกใช้ Ollama")
            response_from_ollama = llama_generate_response(sentence)
            return response_from_ollama



app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'xzxx'
        secret = 'xxxxx'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        response_msg = compute_response(msg)
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
        print(msg, tk)
    except Exception as e:
        print(body)
        print(f"เกิดข้อผิดพลาด: {e}")
    return 'OK'

if __name__ == '__main__':
    # เปิด ngrok tunnel
    port = 5000
    ngrok.set_auth_token("XXXX")  # เปลี่ยนเป็น auth token ของคุณ
    public_url = ngrok.connect(port).public_url
    print(f"ngrok tunnel เปิดที่ {public_url} -> http://127.0.0.1:{port}")
    
    # เริ่มต้น Flask แอป
    app.run(port=port)
