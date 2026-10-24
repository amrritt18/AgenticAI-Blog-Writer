
from __future__ import annotations

import uuid
from datetime import datetime

import streamlit as st

import backend


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.8rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 18px;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }

        .muted {
            opacity: 0.75;
        }

        div[data-testid="stDownloadButton"] button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session state
# --------------------------------------------------
if "blog_result" not in st.session_state:
    st.session_state.blog_result = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "topic" not in st.session_state:
    st.session_state.topic = ""


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("AI Blog Writer")
    st.caption("Powered by LangGraph and multiple LLM providers")

    st.divider()

    st.subheader("Workflow")

    st.markdown(
        """
        1. **Router** — determines whether research is needed.
        2. **Research** — gathers web evidence when needed.
        3. **Planner** — creates the blog outline.
        4. **Workers** — write sections in parallel.
        5. **Reducer** — combines sections into Markdown.
        """
    )

    st.divider()

    st.subheader("LLM providers")
    st.write("1. Groq")
    st.write("2. Google Gemini")
    st.write("3. OpenRouter")

    st.caption(
        "Fallback depends on provider availability, quotas, "
        "and model compatibility."
    )

    st.divider()

    if st.session_state.thread_id:
        st.caption("Latest run ID")
        st.code(st.session_state.thread_id, language=None)

    if st.button("Clear current result", use_container_width=True):
        st.session_state.blog_result = None
        st.session_state.thread_id = None
        st.session_state.topic = ""
        st.rerun()


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>✍️ AI Blog Writer</h1>
        <p class="muted">
            Turn a topic into a structured technical blog using
            a multi-agent workflow, optional web research,
            parallel writing, and automatic LLM fallback.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Blog generation form
# --------------------------------------------------
st.subheader("Create a new blog")

with st.form("blog_generation_form"):
    topic = st.text_area(
        "What would you like to write about?",
        value=st.session_state.topic,
        placeholder=(
            "Example: Predictive maintenance using machine learning "
            "in industrial automation"
        ),
        height=130,
        help="Enter a clear topic for your blog.",
    )

    submitted = st.form_submit_button(
        "Generate blog",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Invoke the LangGraph backend
# --------------------------------------------------
if submitted:
    topic = topic.strip()

    if not topic:
        st.warning("Please enter a blog topic first.")
    else:
        run_id = str(uuid.uuid4())

        st.session_state.topic = topic
        st.session_state.blog_result = None
        st.session_state.thread_id = run_id

        initial_state = {
            "topic": topic,
            "sections": [],
        }

        config = {
            "configurable": {
                "thread_id": run_id,
            }
        }

        try:
            with st.status(
                "Generating your blog...",
                expanded=True,
            ) as status:
                st.write("Starting the multi-agent workflow...")
                st.write("The router will decide whether research is needed.")
                st.write("The planner and writing agents will generate the blog.")
                st.write("The reducer will assemble the final Markdown.")

                result = backend.app.invoke(
                    initial_state,
                    config=config,
                )

                final_blog = result.get("final", "")

                if not isinstance(final_blog, str) or not final_blog.strip():
                    raise RuntimeError(
                        "The workflow completed without returning "
                        "a non-empty final blog."
                    )

                st.session_state.blog_result = final_blog

                status.update(
                    label="Blog generated successfully!",
                    state="complete",
                    expanded=False,
                )

            st.success("Your blog is ready.")

        except Exception as exc:
            st.session_state.blog_result = None

            st.error(
                "Blog generation failed. Check your terminal for "
                "backend errors and verify your API keys, model "
                "availability, Tavily configuration, and database connection."
            )

            with st.expander("Technical error details"):
                st.code(
                    f"{type(exc).__name__}: {exc}",
                    language=None,
                )


# --------------------------------------------------
# Display generated blog
# --------------------------------------------------
final_blog = st.session_state.blog_result

if final_blog:
    st.divider()

    st.subheader("Your generated blog")

    word_count = len(final_blog.split())
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Word count", f"{word_count:,}")
    metric2.metric("Characters", f"{len(final_blog):,}")
    metric3.metric("Format", "Markdown")

    st.caption(f"Generated locally in this session: {generated_at}")

    preview_tab, source_tab = st.tabs(
        ["Blog preview", "Markdown source"]
    )

    with preview_tab:
        st.markdown(final_blog)

    with source_tab:
        st.code(final_blog, language="markdown")

    st.divider()

    safe_topic = "".join(
        char if char.isalnum() else "_"
        for char in st.session_state.topic
    ).strip("_")

    filename = (safe_topic[:80] or "generated_blog") + ".md"

    download_col, info_col = st.columns([1, 2])

    with download_col:
        st.download_button(
            label="Download blog (.md)",
            data=final_blog,
            file_name=filename,
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )

    with info_col:
        st.caption(
            "The downloaded file contains the complete blog in Markdown "
            "format, including headings, links, and code blocks."
        )

else:
    st.info(
        "Enter a topic above and click **Generate blog** to get started."
    )
