# 🚀 GPT-4 Migration Guide

## What Changed?

Your Healthcare AI application has been upgraded from **GPT-2** to **GPT-4** for significantly better medical interpretations and natural language generation.

### Key Improvements

✅ **Better Medical Accuracy** - GPT-4 has much more comprehensive medical knowledge  
✅ **Clearer Explanations** - More coherent and professional clinical interpretations  
✅ **Context Understanding** - Better comprehension of complex medical queries  
✅ **Up-to-date Knowledge** - GPT-4 has more recent medical information  

## Setup Instructions

### Step 1: Get Your OpenAI API Key

1. Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create an OpenAI account
3. Click "Create new secret key"
4. Give it a name (e.g., "Healthcare AI")
5. Copy the key (it starts with `sk-...`)

**Important:** Keep your API key secure and never commit it to version control!

### Step 2: Configure the API Key

#### Option A: Using the Setup Script (Recommended)

```bash
./setup_gpt4.sh
```

This interactive script will:
- Prompt you for your API key
- Set it up for the current session
- Add it to your shell profile for future sessions
- Optionally start the application

#### Option B: Manual Setup

**For current session only:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**For permanent setup (add to ~/.zshrc or ~/.bash_profile):**
```bash
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.zshrc
source ~/.zshrc
```

### Step 3: Install Dependencies

```bash
pip3 install -r requirements.txt
```

The new requirements include the `openai` package instead of `transformers`.

### Step 4: Run the Application

```bash
python3 api.py
```

You should see:
```
✓ GPT-4 API configured (key: ...last8chars)
🩺 Starting Healthcare AI API Server (GPT-4 Powered)...
📊 Dashboard available at: http://localhost:3333
```

## API Changes

### No Breaking Changes!

The API endpoints remain the same:
- `/ask` - General medical queries
- `/classify` - Signal classification with interpretation
- `/feedback` - User feedback
- `/dashboard` - Web interface

### What's Different Under the Hood

**Before (GPT-2):**
- Local model loaded into memory
- Limited medical knowledge
- Slower, less coherent responses
- Required fine-tuning for domain adaptation

**After (GPT-4):**
- API-based (requires internet connection)
- Extensive medical knowledge built-in
- Fast, high-quality responses
- No fine-tuning needed

## Cost Considerations

GPT-4 is a paid API service. Pricing (as of December 2024):

- **GPT-4**: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- **GPT-4 Turbo**: ~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens

**Estimated costs for this application:**
- Average query: ~500 tokens (input + output) = $0.02-0.05 per query
- 100 queries/day ≈ $2-5/day
- 1000 queries/day ≈ $20-50/day

### Cost Optimization Tips

1. **Use GPT-3.5-turbo for lower costs** (change model in `inference.py`)
2. **Implement caching** for common queries
3. **Set usage limits** in your OpenAI account
4. **Monitor usage** via OpenAI dashboard

## Switching to GPT-3.5-turbo (Cheaper Alternative)

If you want to reduce costs, you can use GPT-3.5-turbo instead:

**Edit `inference.py`:**
```python
# Change this line (appears twice in the file):
model="gpt-4",

# To:
model="gpt-3.5-turbo",
```

GPT-3.5-turbo is ~10x cheaper and still much better than GPT-2!

## Troubleshooting

### Error: "OpenAI API key not found"

**Solution:** Make sure you've set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY='your-key-here'
```

### Error: "Rate limit exceeded"

**Solution:** You've hit OpenAI's rate limits. Wait a few minutes or upgrade your OpenAI plan.

### Error: "Insufficient quota"

**Solution:** Add credits to your OpenAI account at [https://platform.openai.com/account/billing](https://platform.openai.com/account/billing)

### Error: "Invalid API key"

**Solution:** Double-check your API key. Make sure there are no extra spaces or quotes.

## Reverting to GPT-2 (If Needed)

If you need to revert to the local GPT-2 model:

```bash
git checkout HEAD~1 inference.py api.py requirements.txt
pip3 install transformers
```

## Testing the Integration

### Test 1: Basic Query
```bash
curl -X POST http://localhost:3333/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain atrial fibrillation"}'
```

### Test 2: Signal Classification
```bash
curl -X POST http://localhost:3333/classify \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.1, 0.2, 0.15, 0.18, 0.22],
    "signal_type": "ECG",
    "clinical_context": "Patient with palpitations"
  }'
```

You should see much more detailed and accurate responses compared to GPT-2!

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Rotate keys regularly** (every 90 days)
4. **Set spending limits** in OpenAI dashboard
5. **Monitor usage** for unexpected spikes
6. **Use separate keys** for dev/staging/production

## Support

- **OpenAI Documentation:** [https://platform.openai.com/docs](https://platform.openai.com/docs)
- **OpenAI API Status:** [https://status.openai.com](https://status.openai.com)
- **Pricing:** [https://openai.com/pricing](https://openai.com/pricing)

## Summary

✅ Application upgraded to GPT-4  
✅ Better medical interpretations  
✅ Same API endpoints (no breaking changes)  
✅ Requires OpenAI API key  
✅ Internet connection required  
✅ Pay-per-use pricing  

**Next Steps:**
1. Get your OpenAI API key
2. Run `./setup_gpt4.sh` or set `OPENAI_API_KEY` manually
3. Start the application with `python3 api.py`
4. Test the improved responses!

Enjoy your upgraded Healthcare AI system! 🎉
