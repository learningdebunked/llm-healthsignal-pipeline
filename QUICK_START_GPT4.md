# ⚡ Quick Start - GPT-4 Version

## 1. Get API Key
Visit: https://platform.openai.com/api-keys

## 2. Set API Key
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

## 3. Install & Run
```bash
pip3 install -r requirements.txt
python3 api.py
```

## 4. Access
Open: http://localhost:3333

---

## Alternative: Use Setup Script
```bash
./setup_gpt4.sh
```

---

## Test It
```bash
curl -X POST http://localhost:3333/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is atrial fibrillation?"}'
```

---

## Cost
- ~$0.02-0.05 per query
- Monitor at: https://platform.openai.com/usage

---

## Need Help?
See: GPT4_MIGRATION_GUIDE.md
