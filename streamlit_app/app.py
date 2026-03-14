from pathlib import Path
import sys

import httpx
import streamlit as st

# Streamlit Cloud may execute this file with only `streamlit_app/` on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings

settings = get_settings()

st.set_page_config(page_title="Insurance Policy Chatbot", page_icon="🩺", layout="wide")
st.title("Insurance Policy Chatbot")
st.caption("Policy Q&A with ingestion controls and cited answers.")

backend_url = settings.streamlit_backend_url.rstrip("/")
st.write(f"Backend URL: `{backend_url}`")


def check_backend_health(base_url: str) -> tuple[bool, dict]:
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{base_url}/health")
            response.raise_for_status()
            return True, response.json()
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


def fetch_ingest_status(base_url: str) -> tuple[bool, dict]:
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(f"{base_url}/ingest/status")
            response.raise_for_status()
            return True, response.json()
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


def trigger_ingest(base_url: str, seed_url: str | None = None) -> tuple[bool, dict]:
    payload = {"mode": "incremental"}
    if seed_url:
        payload["seed_url"] = seed_url.strip()
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{base_url}/ingest", json=payload)
            response.raise_for_status()
            return True, response.json()
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


def ask_question(base_url: str, question: str, top_k: int) -> tuple[bool, dict]:
    payload = {"question": question, "top_k": top_k}
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{base_url}/chat", json=payload)
            response.raise_for_status()
            return True, response.json()
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


def verify_admin_password(entered_password: str) -> bool:
    configured = settings.admin_password.strip()
    if not configured:
        st.error("ADMIN_PASSWORD is not configured.")
        return False
    if entered_password != configured:
        st.error("Invalid admin password.")
        return False
    return True


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "admin_password_input" not in st.session_state:
    st.session_state.admin_password_input = ""


admin_col, chat_col = st.columns([1, 2], gap="large")

with admin_col:
    st.subheader("Admin")
    entered_admin_password = st.text_input(
        "Admin password (required for each admin action)",
        type="password",
        key="admin_password_input",
    )

    with st.expander("Backend Health", expanded=True):
        if st.button("Check Backend Health", use_container_width=True):
            allowed = verify_admin_password(entered_admin_password)
            st.session_state.admin_password_input = ""
            if allowed:
                ok, payload = check_backend_health(backend_url)
                if ok:
                    st.success("Backend is reachable.")
                else:
                    st.error("Backend health check failed.")
                st.json(payload)

    with st.expander("Ingestion", expanded=True):
        seed_url = st.text_input(
            "Optional seed URL override",
            value=settings.policy_seed_url,
            help="Leave as default unless you want a different policy entry page.",
        )
        if st.button("Run Incremental Re-Ingest", use_container_width=True):
            allowed = verify_admin_password(entered_admin_password)
            st.session_state.admin_password_input = ""
            if allowed:
                ok, payload = trigger_ingest(backend_url, seed_url=seed_url)
                if ok:
                    st.success("Ingestion request completed.")
                else:
                    st.error("Ingestion request failed.")
                st.json(payload)

        if st.button("Refresh Ingestion Status", use_container_width=True):
            allowed = verify_admin_password(entered_admin_password)
            st.session_state.admin_password_input = ""
            if allowed:
                ok, payload = fetch_ingest_status(backend_url)
                if ok:
                    st.info(f"Status: {payload.get('status', 'unknown')}")
                else:
                    st.error("Could not fetch ingestion status.")
                st.json(payload)

with chat_col:
    st.subheader("Policy Chat")
    top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=6)

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a policy question",
            placeholder="Example: What prior authorization rules apply for treatment X?",
            height=120,
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        ok, payload = ask_question(backend_url, question.strip(), top_k=top_k)
        if ok:
            st.session_state.chat_history.append(
                {
                    "question": question.strip(),
                    "response": payload,
                }
            )
        else:
            st.error("Chat request failed.")
            st.json(payload)

    for item in reversed(st.session_state.chat_history):
        st.markdown("---")
        st.markdown(f"**You:** {item['question']}")
        response = item["response"]
        answer = response.get("answer", "")
        is_refusal = bool(response.get("is_refusal", True))
        citations = response.get("citations", [])

        if is_refusal:
            st.warning(answer or "I don't know based on the provided policy documents.")
        else:
            st.success(answer)

        if not is_refusal:
            st.markdown("**Citations**")
            if citations:
                for citation in citations:
                    source_url = citation.get("source_url", "")
                    section = citation.get("section") or "Section not specified"
                    excerpt = citation.get("excerpt", "")
                    st.markdown(f"- [{section}]({source_url})")
                    if excerpt:
                        st.caption(excerpt)
            else:
                st.error("Answer returned without citations.")

        debug_scores = (response.get("retrieval_debug") or {}).get("top_k_scores", [])
        if debug_scores:
            st.caption(f"Retrieval scores: {debug_scores}")
