# LLM Selection Analysis for Phishing Explanation Module

## 1. Objective
To integrate a Large Language Model (LLM) that translates technical machine learning features (SHAP values, confidence scores, modality probabilities) into concise, non-technical natural language explanations for the end-user.

## 2. Recommended Models

For a real-time web scanning application (FYP), **Latency** and **Cost** are the primary constraints.

### Option A: Gemini 1.5 Flash (Google) — **Top Recommendation**
*   **Type:** Cloud API
*   **Pros:**
    *   **Speed:** Designed specifically for high-volume, low-latency tasks. Fastest in its class.
    *   **Cost:** Generous free tier for development/prototyping. Very cheap at scale.
    *   **Context:** Large context window allows you to pass detailed JSON dumps without truncation issues.
*   **Cons:** Requires an API key (Data leaves local machine).

### Option B: GPT-4o-mini (OpenAI)
*   **Type:** Cloud API
*   **Pros:**
    *   **Standardization:** Industry standard API, easy to integrate.
    *   **Quality:** Very reliable instruction following.
*   **Cons:** No free tier (requires credit card setup), though extremely cheap.

### Option C: Llama 3 (via Ollama) — **Privacy Recommendation**
*   **Type:** Local Execution
*   **Pros:**
    *   **Privacy:** **Zero data leakage**. URL data never leaves the scanning server.
    *   **Cost:** Free (uses local hardware).
*   **Cons:**
    *   **Hardware:** Requires decent RAM/GPU.
    *   **Setup:** Adds complexity (installing Ollama).
    *   **Latency:** Slower than cloud APIs unless running on a powerful GPU.

## 3. Comparison Matrix

| Feature | Gemini 1.5 Flash | GPT-4o-mini | Llama 3 (Local) |
| :--- | :--- | :--- | :--- |
| **Speed (Latency)** | ***** (Fastest) | ***** (Fast) | ** (Hardware dependent) |
| **Cost** | ***** (Free Tier) | **** (Low Cost) | ***** (Free) |
| **Implementation** | **** (Easy) | **** (Easy) | ** (Medium - Setup req.) |
| **Data Privacy** | *** (Cloud) | *** (Cloud) | ***** (Local) |
| **Accuracy** | **** | **** | *** |

## 4. Recommendation

**For an FYP submission:**
I recommend **Gemini 1.5 Flash**.
1.  **Free to implement**: You don't need to add a credit card.
2.  **Fast**: It won't slow down your demo during the presentation.
3.  **Performance**: It is more than capable of handling this specific "data-to-text" summarization task.
