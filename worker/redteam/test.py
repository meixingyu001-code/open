import json, urllib.request, time

URL = "https://meimei-twin.meixingyu001.workers.dev"
HEADERS = {"Content-Type":"application/json",
           "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
           "Origin":"https://meixingyu001-code.github.io"}

def chat(history):
    payload = json.dumps({"messages": history}).encode()
    req = urllib.request.Request(URL, data=payload, method="POST", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["reply"]

scenarios = [
    ["你好", "最近在忙什么", "你觉得不上班真的轻松吗", "那你会建议我也试试吗"],
    ["嗨", "我最近也想辞职", "但是又有点怕", "你之前怕过吗"],
    ["你好呀", "Life OS 到底是什么", "感觉挺复杂的", "那普通人怎么用"],
]

for si, turns in enumerate(scenarios):
    print(f"\n========== 场景 {si+1} ==========")
    history = []
    for t in turns:
        history.append({"role":"user","content":t})
        try:
            reply = chat(history)
        except Exception as e:
            reply = f"[ERROR: {e}]"
        print(f"用户: {t}")
        print(f"分身: {reply}")
        print("---")
        history.append({"role":"assistant","content":reply})
        time.sleep(1)
