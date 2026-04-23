import os
import json
import re
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Step 1 - LOAD KNOWLEDGE BASE (RAG)

def load_knowledge_base(path: str = "knowledge_base/autostream_kb.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)

KB = load_knowledge_base()


def rag_retrieve(query: str) -> str:
    query_lower = query.lower()
    context_parts = []

    # Pricing / plan keywords
    if any(kw in query_lower for kw in ["price", "pricing", "plan", "cost", "how much", "basic", "pro", "subscription"]):
        plans = KB["pricing"]["plans"]
        context_parts.append("=== AutoStream Pricing Plans ===")
        for plan in plans:
            context_parts.append(
                f"\n{plan['name']} - {plan['price']}\n"
                f"Features: {', '.join(plan['features'])}\n"
                f"Best for: {plan['best_for']}"
            )

    # Feature keywords
    if any(kw in query_lower for kw in ["feature", "caption", "4k", "edit", "template", "analytic", "render"]):
        context_parts.append("\n=== AutoStream Features ===")
        for feature, desc in KB["features"].items():
            context_parts.append(f"- {feature.replace('_', ' ').title()}: {desc}")

    # Policy keywords
    if any(kw in query_lower for kw in ["refund", "cancel", "policy", "support", "trial", "free"]):
        context_parts.append("\n=== Company Policies ===")
        context_parts.append(f"Refund Policy: {KB['policies']['refund_policy']}")
        context_parts.append(f"Support (Basic): {KB['policies']['support']['basic']}")
        context_parts.append(f"Support (Pro): {KB['policies']['support']['pro']}")
        context_parts.append(f"Free Trial: {KB['policies']['trial']}")
        context_parts.append(f"Cancellation: {KB['policies']['cancellation']}")

    # Platform keywords
    if any(kw in query_lower for kw in ["youtube", "instagram", "tiktok", "platform", "channel"]):
        context_parts.append(f"\n=== Supported Platforms ===")
        context_parts.append(f"AutoStream supports: {', '.join(KB['platforms_supported'])}")

    # General company info fallback
    if not context_parts:
        context_parts.append(f"=== About AutoStream ===")
        context_parts.append(f"{KB['company']['description']}")
        context_parts.append("\nAvailable plans: Basic ($29/mo) and Pro ($79/mo)")

    return "\n".join(context_parts)


# Step 2 - MOCK LEAD CAPTURE TOOL

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API call to capture lead. Prints confirmation and returns success message."""
    print(f"\n{'='*50}")
    print(f"LEAD CAPTURED SUCCESSFULLY!")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Platform: {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# Step 3 - LANGGRAPH STATE

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]   # Full conversation history
    intent: Optional[str]                      # casual | product_inquiry | high_intent
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool                      # True when in lead collection mode
    current_field: Optional[str]              # name | email | platform


# Step 4 - LLM SETUP (Gemini free tier)

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Then run: export GEMINI_API_KEY='your_key_here'"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )


# Step 5 - INTENT CLASSIFICATION NODE

def classify_intent(state: AgentState) -> AgentState:
    llm = get_llm()
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    classification_prompt = f"""You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the user message into EXACTLY ONE of these intents:
- "casual_greeting": Simple greetings, small talk, non-specific openers
- "product_inquiry": Questions about features, pricing, plans, policies, comparisons
- "high_intent": User shows clear desire to sign up, buy, try, or start using the product

User message: "{last_human_msg}"

Respond with ONLY the intent label, nothing else. Examples:
- "Hi there!" → casual_greeting
- "What's the difference between Basic and Pro?" → product_inquiry  
- "I want to sign up for the Pro plan" → high_intent
- "This looks great, I need this for my YouTube channel" → high_intent
- "How much does it cost?" → product_inquiry
"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    intent = response.content.strip().lower().replace('"', '').replace("'", "")

    # Normalize
    if "high" in intent:
        intent = "high_intent"
    elif "product" in intent or "inquiry" in intent:
        intent = "product_inquiry"
    else:
        intent = "casual_greeting"

    return {**state, "intent": intent}


# Step 6 - RESPONSE GENERATION NODE

def generate_response(state: AgentState) -> AgentState:
    """Generate agent response based on intent and state."""
    llm = get_llm()

    # ── If we're mid-lead-collection, handle field extraction ──
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return handle_lead_collection(state, llm)

    intent = state.get("intent", "casual_greeting")
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    # Retrieve relevant context from knowledge base
    rag_context = rag_retrieve(last_human_msg)

    # Build system prompt
    system_prompt = f"""You are Alex, a friendly and knowledgeable sales assistant for AutoStream — an AI-powered video editing SaaS for content creators.

KNOWLEDGE BASE CONTEXT:
{rag_context}

GUIDELINES:
- Be warm, concise, and helpful
- Use the knowledge base context to answer accurately
- If asked about pricing, always mention both plans clearly
- Never make up features or prices not in the context
- Keep responses under 150 words unless detail is needed
- If intent is high_intent, after answering, express enthusiasm and smoothly transition to collecting their info
"""

    # Build messages for the LLM
    llm_messages = [SystemMessage(content=system_prompt)]

    # Add conversation history (last 8 messages for context window)
    for msg in state["messages"][-8:]:
        llm_messages.append(msg)

    response = llm.invoke(llm_messages)
    ai_reply = response.content.strip()

    new_state = {**state}

    # If high intent and not yet collecting, start lead collection
    if intent == "high_intent" and not state.get("collecting_lead") and not state.get("lead_captured"):
        ai_reply += "\n\nI'd love to get you started right away! Could I grab your name first?"
        new_state["collecting_lead"] = True
        new_state["current_field"] = "name"

    new_state["messages"] = state["messages"] + [AIMessage(content=ai_reply)]
    return new_state


def handle_lead_collection(state: AgentState, llm) -> AgentState:
    """Handle the multi-turn lead data collection flow."""
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    new_state = {**state}
    current_field = state.get("current_field")
    ai_reply = ""

    # Extract value for the current field
    if current_field == "name":
        new_state["lead_name"] = last_human_msg.strip()
        ai_reply = f"Nice to meet you, {new_state['lead_name']}! What's your email address?"
        new_state["current_field"] = "email"

    elif current_field == "email":
        # Basic email validation
        email = last_human_msg.strip()
        if re.match(r"[^@]+@[^@]+\.[^@]+", email):
            new_state["lead_email"] = email
            ai_reply = f"Got it! And which platform do you create content for? (e.g., YouTube, Instagram, TikTok)"
            new_state["current_field"] = "platform"
        else:
            ai_reply = "Hmm, that doesn't look like a valid email. Could you double-check and re-enter your email address?"

    elif current_field == "platform":
        new_state["lead_platform"] = last_human_msg.strip()

        # ALL THREE FIELDS COLLECTED — FIRE THE TOOL
        result = mock_lead_capture(
            name=new_state["lead_name"],
            email=new_state["lead_email"],
            platform=new_state["lead_platform"]
        )

        ai_reply = (
            f"You're all set, {new_state['lead_name']}!\n\n"
            f"We've captured your details and our team will reach out to {new_state['lead_email']} shortly "
            f"to help you get started with AutoStream for your **{new_state['lead_platform']}** content.\n\n"
            f"In the meantime, feel free to ask me anything about AutoStream! 😊"
        )
        new_state["lead_captured"] = True
        new_state["collecting_lead"] = False
        new_state["current_field"] = None

    new_state["messages"] = state["messages"] + [AIMessage(content=ai_reply)]
    return new_state


# Step 7 - ROUTING LOGIC

def route_after_classify(state: AgentState) -> str:
    return "generate_response"


# Step 8 - BUILD LANGGRAPH

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"generate_response": "generate_response"}
    )
    graph.add_edge("generate_response", END)

    return graph.compile()


# Step 9 - MAIN CONVERSATION LOOP

def run_agent():
    print("\n" + "-"*50)
    print("AutoStream AI Assistant")
    print("-"*50)

    graph = build_graph()

    # Initial state
    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
        "current_field": None,
    }

    # Opening message from agent
    opening = "Hi there! I'm Alex from AutoStream. How can I help you today?"
    print(f"🤖 Alex: {opening}\n")
    state["messages"].append(AIMessage(content=opening))

    while True:
        user_input = input("👤 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\n🤖 Alex: Thanks for chatting! Have a great day!")
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        try:
            result = graph.invoke(state)
            state = result

            # Print agent response
            last_ai_msg = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai_msg = msg.content
                    break

            print(f"\n🤖 Alex: {last_ai_msg}\n")

        except Exception as e:
            print(f"\nError: {e}\n")
            if "GEMINI_API_KEY" in str(e):
                print("Please set your GEMINI_API_KEY environment variable.")
                print("Get a free key at: https://aistudio.google.com/app/apikey")
                break


if __name__ == "__main__":
    run_agent()
