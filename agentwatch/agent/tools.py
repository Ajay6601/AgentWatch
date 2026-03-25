"""
Tools available to the support agent.
Kept simple — the agent is just a data source for realistic traces.
"""

from langchain_core.tools import tool

# Simulated knowledge base articles
KB_ARTICLES = {
    "password_reset": {
        "title": "How to Reset Your Password",
        "content": "Go to Settings > Security > Reset Password. Enter your current password, "
        "then create a new one with at least 12 characters including uppercase, lowercase, "
        "number, and special character. You'll receive a confirmation email within 5 minutes.",
        "category": "account",
    },
    "billing_dispute": {
        "title": "Disputing a Charge",
        "content": "Navigate to Billing > Transaction History. Find the charge and click "
        "'Dispute'. Provide a reason and any supporting documentation. Our team reviews "
        "disputes within 3-5 business days. Refunds appear in 7-10 business days.",
        "category": "billing",
    },
    "api_rate_limits": {
        "title": "API Rate Limits",
        "content": "Free tier: 100 requests/minute. Pro: 1000 req/min. Enterprise: custom. "
        "Rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset. "
        "When exceeded, you'll receive HTTP 429 with a Retry-After header.",
        "category": "technical",
    },
    "integration_setup": {
        "title": "Setting Up Integrations",
        "content": "Go to Settings > Integrations. Click 'Add New'. Select the service "
        "(Slack, GitHub, Jira supported). Follow the OAuth flow. Configure webhook URL "
        "for real-time events. Test with the 'Send Test Event' button.",
        "category": "technical",
    },
    "data_export": {
        "title": "Exporting Your Data",
        "content": "Go to Settings > Data > Export. Choose format (CSV, JSON, Parquet). "
        "Select date range. Large exports (>1GB) are processed async — you'll get an "
        "email with download link within 1 hour. GDPR exports include all PII fields.",
        "category": "data",
    },
    "team_permissions": {
        "title": "Managing Team Permissions",
        "content": "Admins can manage roles at Settings > Team. Roles: Viewer (read-only), "
        "Editor (create/edit), Admin (full access including billing). SSO available on "
        "Enterprise plan. SCIM provisioning for automatic user management.",
        "category": "account",
    },
}


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant articles about a customer issue."""
    query_lower = query.lower()
    results = []
    for key, article in KB_ARTICLES.items():
        score = sum(1 for word in query_lower.split() if word in key or word in article["content"].lower())
        if score > 0:
            results.append((score, article))
    results.sort(key=lambda x: x[0], reverse=True)
    if not results:
        return "No relevant articles found."
    top = results[0][1]
    return f"**{top['title']}**\n{top['content']}"


@tool
def lookup_customer(customer_id: str) -> str:
    """Look up customer account information by their ID."""
    # Simulated customer data
    customers = {
        "C001": {"name": "Acme Corp", "plan": "Enterprise", "status": "active", "mrr": "$2,400"},
        "C002": {"name": "StartupXYZ", "plan": "Pro", "status": "active", "mrr": "$99"},
        "C003": {"name": "BigRetail Inc", "plan": "Enterprise", "status": "churned", "mrr": "$0"},
        "C004": {"name": "DevShop LLC", "plan": "Free", "status": "active", "mrr": "$0"},
        "C005": {"name": "MegaFinance", "plan": "Enterprise", "status": "active", "mrr": "$5,000"},
    }
    c = customers.get(customer_id)
    if not c:
        return f"Customer {customer_id} not found."
    return f"Customer: {c['name']} | Plan: {c['plan']} | Status: {c['status']} | MRR: {c['mrr']}"


@tool
def check_system_status(service: str) -> str:
    """Check the current operational status of a service component."""
    statuses = {
        "api": "operational — 99.98% uptime last 30d, avg latency 45ms",
        "database": "operational — 99.99% uptime, replication lag <1ms",
        "auth": "degraded — elevated error rates since 14:30 UTC, team investigating",
        "webhooks": "operational — delivery rate 99.7%, avg delay 230ms",
        "search": "operational — index freshness <30s",
    }
    s = statuses.get(service.lower())
    if not s:
        return f"Unknown service: {service}. Available: {', '.join(statuses.keys())}"
    return f"Service '{service}': {s}"


@tool
def create_ticket(summary: str, priority: str = "medium") -> str:
    """Create an internal support ticket for escalation."""
    import random

    ticket_id = f"TK-{random.randint(10000, 99999)}"
    return f"Ticket {ticket_id} created: '{summary}' (priority: {priority}). Assigned to support queue."


AGENT_TOOLS = [search_knowledge_base, lookup_customer, check_system_status, create_ticket]

