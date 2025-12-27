# 📋 GPT-4 Setup Checklist

Use this checklist to ensure your Healthcare AI application is properly configured with GPT-4.

---

## ✅ Pre-Setup

- [ ] Python 3.8+ installed
  ```bash
  python3 --version
  ```

- [ ] pip/pip3 available
  ```bash
  pip3 --version
  ```

- [ ] Internet connection available
  ```bash
  ping -c 3 api.openai.com
  ```

- [ ] Git repository cloned (if applicable)
  ```bash
  cd llm-healthsignal-pipeline
  ```

---

## ✅ OpenAI Account Setup

- [ ] OpenAI account created
  - Visit: https://platform.openai.com/signup

- [ ] API key generated
  - Visit: https://platform.openai.com/api-keys
  - Click "Create new secret key"
  - Copy key (starts with `sk-...`)

- [ ] Billing configured (if needed)
  - Visit: https://platform.openai.com/account/billing
  - Add payment method
  - Set usage limits (recommended)

- [ ] API key saved securely
  - [ ] Stored in password manager
  - [ ] NOT committed to git
  - [ ] NOT shared publicly

---

## ✅ Application Setup

- [ ] Dependencies installed
  ```bash
  pip3 install -r requirements.txt
  ```

- [ ] OpenAI package verified
  ```bash
  python3 -c "import openai; print(openai.__version__)"
  ```

- [ ] API key configured

  **Option A: Interactive setup**
  ```bash
  ./setup_gpt4.sh
  ```

  **Option B: Manual setup**
  ```bash
  export OPENAI_API_KEY='sk-your-key-here'
  ```

  **Option C: Shell profile (permanent)**
  ```bash
  echo "export OPENAI_API_KEY='sk-your-key-here'" >> ~/.zshrc
  source ~/.zshrc
  ```

- [ ] API key verified
  ```bash
  echo $OPENAI_API_KEY
  # Should show: sk-...
  ```

---

## ✅ Application Testing

- [ ] Application starts successfully
  ```bash
  python3 api.py
  ```

- [ ] Startup message shows GPT-4 configured
  ```
  ✓ GPT-4 API configured (key: ...last8chars)
  🩺 Starting Healthcare AI API Server (GPT-4 Powered)...
  ```

- [ ] Server accessible in browser
  - Open: http://localhost:3333
  - Should see API info or dashboard

- [ ] API endpoints responding

  **Test /ask endpoint:**
  ```bash
  curl -X POST http://localhost:3333/ask \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is atrial fibrillation?"}'
  ```

  **Test /classify endpoint:**
  ```bash
  curl -X POST http://localhost:3333/classify \
    -H "Content-Type: application/json" \
    -d '{"signal": [0.1, 0.2, 0.15], "signal_type": "ECG"}'
  ```

- [ ] Responses are detailed and medical (not generic)
  - Should see professional medical explanations
  - Should include clinical details
  - Should be structured and comprehensive

- [ ] Test script runs successfully
  ```bash
  python3 test_gpt4_api.py
  ```

---

## ✅ Configuration Verification

- [ ] Environment variables set correctly
  ```bash
  env | grep OPENAI
  ```

- [ ] No API key in code files
  ```bash
  grep -r "sk-" *.py
  # Should return nothing
  ```

- [ ] .gitignore includes sensitive files
  ```bash
  cat .gitignore | grep -E "\.env|\.key"
  ```

- [ ] API key works with OpenAI
  ```bash
  curl https://api.openai.com/v1/models \
    -H "Authorization: Bearer $OPENAI_API_KEY"
  ```

---

## ✅ Documentation Review

- [ ] Read QUICK_START_GPT4.md
- [ ] Read GPT4_MIGRATION_GUIDE.md
- [ ] Understand cost implications
- [ ] Know how to monitor usage
- [ ] Familiar with troubleshooting steps

---

## ✅ Security Checklist

- [ ] API key stored in environment variable (not hardcoded)
- [ ] .env file in .gitignore
- [ ] API key not in git history
  ```bash
  git log --all --full-history --source -- "*" | grep -i "sk-"
  # Should return nothing
  ```
- [ ] Usage limits set in OpenAI dashboard
- [ ] Billing alerts configured
- [ ] API key rotation schedule planned (every 90 days)

---

## ✅ Production Readiness (Optional)

- [ ] Error handling tested
  - [ ] Invalid API key
  - [ ] Network timeout
  - [ ] Rate limit exceeded
  - [ ] Insufficient quota

- [ ] Logging configured
  - [ ] API calls logged
  - [ ] Errors logged
  - [ ] Usage tracked

- [ ] Monitoring setup
  - [ ] OpenAI usage dashboard checked
  - [ ] Cost alerts configured
  - [ ] Performance metrics tracked

- [ ] Backup plan
  - [ ] Fallback responses working
  - [ ] Alternative API key available
  - [ ] Downtime handling strategy

---

## ✅ Cost Management

- [ ] Understand pricing model
  - GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens
  - Average query: ~$0.02-0.05

- [ ] Usage limits set
  - Visit: https://platform.openai.com/account/limits
  - Set monthly budget cap

- [ ] Billing alerts configured
  - Visit: https://platform.openai.com/account/billing
  - Set alert thresholds

- [ ] Usage monitoring plan
  - Check dashboard weekly
  - Review costs monthly
  - Optimize if needed

---

## ✅ Optional Optimizations

- [ ] Consider GPT-3.5-turbo for cost savings
  - Edit `inference.py`
  - Change `model="gpt-4"` to `model="gpt-3.5-turbo"`
  - 10x cheaper, still better than GPT-2

- [ ] Implement response caching
  - Cache common queries
  - Reduce API calls
  - Save costs

- [ ] Add rate limiting
  - Prevent abuse
  - Control costs
  - Protect API key

- [ ] Set up analytics
  - Track query types
  - Monitor response quality
  - Identify optimization opportunities

---

## 🎯 Final Verification

Run this complete test sequence:

```bash
# 1. Check environment
echo "Checking environment..."
python3 --version
pip3 --version
echo $OPENAI_API_KEY | cut -c1-10

# 2. Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# 3. Start server (in background)
echo "Starting server..."
python3 api.py &
SERVER_PID=$!
sleep 5

# 4. Test API
echo "Testing API..."
curl -X POST http://localhost:3333/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test query"}' \
  | python3 -m json.tool

# 5. Stop server
echo "Stopping server..."
kill $SERVER_PID

echo "✅ All checks complete!"
```

---

## 📊 Success Criteria

Your setup is successful if:

✅ Server starts without errors  
✅ GPT-4 configuration message appears  
✅ API endpoints return detailed medical responses  
✅ No API key errors  
✅ Responses are professional and comprehensive  
✅ Test script passes all tests  

---

## 🆘 Troubleshooting

If any check fails, see:

- **QUICK_START_GPT4.md** - Quick setup guide
- **GPT4_MIGRATION_GUIDE.md** - Detailed troubleshooting
- **README.md** - General documentation

Common issues:

| Issue | Solution |
|-------|----------|
| "API key not found" | Set OPENAI_API_KEY environment variable |
| "Invalid API key" | Check key is correct, no extra spaces |
| "Rate limit exceeded" | Wait a few minutes, check usage limits |
| "Insufficient quota" | Add credits to OpenAI account |
| Generic responses | Verify GPT-4 is being used, not fallback |

---

## 📞 Support Resources

- **OpenAI Status:** https://status.openai.com
- **OpenAI Docs:** https://platform.openai.com/docs
- **Usage Dashboard:** https://platform.openai.com/usage
- **Billing:** https://platform.openai.com/account/billing

---

## ✨ You're Ready!

Once all checkboxes are complete, your Healthcare AI application is fully configured with GPT-4 and ready for use!

**Start the application:**
```bash
python3 api.py
```

**Access the dashboard:**
```
http://localhost:3333
```

**Enjoy superior medical interpretations powered by GPT-4!** 🩺🤖✨

---

**Last Updated:** December 27, 2024  
**Version:** 2.0
