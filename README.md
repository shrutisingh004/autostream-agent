# 🎬 AutoStream AI Sales Agent

## Overview

AutoStream's AI agent is a conversational sales assistant that:
- Answers product/pricing questions using a **local RAG knowledge base**
- **Classifies user intent** (greeting / product inquiry / high-intent lead)
- **Collects lead details** (name, email, platform) when a user shows buying intent
- **Fires a mock lead capture tool** only after all three fields are collected

---

## Project Structure

```
autostream-agent/
├── agent.py                        # Core agent logic (LangGraph + Gemini)
├── knowledge_base/
│   └── autostream_kb.json          # Local RAG knowledge base
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### Step 1 — Clone / Download

```bash
git clone https://github.com/shrutisingh004/autostream-agent.git
cd autostream-agent
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate       # Linux : source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Agent

```bash
python agent.py
```

---

## Example Conversation

```
🤖 Alex: Hi there! I'm Alex from AutoStream. How can I help you today?

👤 You: Hi, tell me about your pricing

🤖 Alex: We have two plans at AutoStream:

**Basic Plan - $29/month**
- 10 videos/month, 720p resolution, standard templates, email support

**Pro Plan - $79/month**
- Unlimited videos, 4K resolution, AI captions, 24/7 support, analytics

Which plan sounds right for your needs?

👤 You: I want to try the Pro plan for my YouTube channel

🤖 Alex: That's awesome — Pro is perfect for YouTube creators!
I'd love to get you started. Could I grab your name first?

👤 You: John Doe

🤖 Alex: Nice to meet you, John Doe! What's your email address?

👤 You: john@example.com

🤖 Alex: Got it! And which platform do you create content for?

👤 You: YouTube

==================================================
🎯 LEAD CAPTURED SUCCESSFULLY!
   Name     : John Doe
   Email    : john@example.com
   Platform : YouTube
==================================================

🤖 Alex: You're all set, John Doe! We'll reach out to john@example.com shortly.
```

---

## Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, deterministic state management** through a typed state graph — critical for a lead-capture workflow where premature tool triggering must be prevented. Unlike AutoGen's agent-to-agent messaging model, LangGraph gives fine-grained control over **when** transitions happen and **what** data persists.

### How State is Managed

The `AgentState` TypedDict holds the complete conversation state across all turns:

- **`messages`**: Full conversation history (accumulated via LangGraph's `add_messages` reducer), giving the LLM memory across 5–6+ turns
- **`intent`**: Updated every turn by the intent classifier node
- **`collecting_lead`** + **`current_field`**: State machine flags that track which lead field to collect next (`name → email → platform`)
- **`lead_name / email / platform`**: Persisted across turns until all three are collected
- **`lead_captured`**: Prevents re-triggering the tool

### Flow

```
User Message → [classify_intent node] → [generate_response node] → END
                      ↓                          ↓
               Sets intent field         Uses RAG + intent to reply
                                         Triggers lead tool only when
                                         all 3 fields are collected
```

### RAG Pipeline

The RAG layer is a lightweight keyword-matching retriever over a local JSON knowledge base. When a user asks about pricing, features, or policies, the relevant section is extracted and injected into the LLM's system prompt as context. No vector database is needed for this scale — the structured JSON enables precise, fast retrieval.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business API (via Meta Cloud API)**:

### Architecture

```
WhatsApp User
     ↓  (sends message)
Meta WhatsApp Cloud API
     ↓  (POST webhook)
Your Server (FastAPI / Flask)
     ↓  (invokes)
AutoStream LangGraph Agent
     ↓  (returns reply)
Meta API → WhatsApp User
```

### Step-by-Step Integration

1. **Register on Meta for Developers** at [developers.facebook.com](https://developers.facebook.com)
2. **Create a WhatsApp Business App** and get a phone number + API token
3. **Set up a Webhook endpoint** using FastAPI:

```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
graph = build_graph()
sessions = {}  # In-memory session store per phone number

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    body = await request.json()
    
    # Parse incoming message
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone = message["from"]
    text = message["text"]["body"]
    
    # Get or create session state for this user
    if phone not in sessions:
        sessions[phone] = {
            "messages": [], "intent": None,
            "lead_name": None, "lead_email": None,
            "lead_platform": None, "lead_captured": False,
            "collecting_lead": False, "current_field": None
        }
    
    state = sessions[phone]
    state["messages"].append(HumanMessage(content=text))
    
    # Run agent
    result = graph.invoke(state)
    sessions[phone] = result
    
    # Extract reply
    reply = [m for m in result["messages"] if isinstance(m, AIMessage)][-1].content
    
    # Send reply back via WhatsApp API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v18.0/{PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": phone,
                "text": {"body": reply}
            }
        )
    return {"status": "ok"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    # WhatsApp webhook verification
    params = request.query_params
    if params.get("hub.verify_token") == YOUR_VERIFY_TOKEN:
        return int(params.get("hub.challenge"))
```

4. **Deploy to a public server** (Railway, Render, AWS EC2) — WhatsApp requires HTTPS
5. **Register your webhook URL** in the Meta Developer Console
6. **Session persistence**: For production, replace the in-memory `sessions` dict with Redis to survive server restarts

### Key Considerations
- Each WhatsApp phone number = independent `AgentState` session
- WhatsApp has a **24-hour messaging window** — the agent must respond within 24h of last user message
- Use **message templates** for initiating conversations (outbound lead follow-ups)
- Add **rate limiting** to avoid exceeding Meta API quotas

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent state graph orchestration |
| `langchain` | LLM abstractions, message types |
| `langchain-google-genai` | Gemini 1.5 Flash integration |
| `google-generativeai` | Google AI SDK |

---

## Running Tests (Manual)

Test the full happy path by entering these messages in sequence:
1. `"Hi there"` → Should get a greeting response
2. `"What are your pricing plans?"` → Should retrieve pricing from KB
3. `"I want to sign up for Pro for my Instagram"` → Should detect high intent
4. Enter your name when asked
5. Enter your email when asked
6. Enter your platform → Should trigger `mock_lead_capture()`

---
<div align="center">
*Built for ServiceHive's Inflx platform assignment — AutoStream Social-to-Lead Agent*
</div>
