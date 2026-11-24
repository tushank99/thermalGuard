"""Score the assistant with RAGAS (faithfulness, answer relevancy, context precision) on a golden set.

Judge LLM = the same free-tier LLM; embeddings = local MiniLM. Writes reports/ragas.json.
Usage:  python -m thermalguard.rag.evaluate
"""
from __future__ import annotations

import json
import os
import time
import uuid

from thermalguard import config
from thermalguard.rag.graph import ask

GOLDEN = [
    ("What does sensor_11 measure?", "sensor_11 is Ps30, the static pressure at the HPC outlet, in psia."),
    ("Which sensors carry no degradation information under a single operating condition?",
     "Sensors 1, 5, 6, 10, 16, 18 and 19 are effectively constant and carry no degradation information."),
    ("What are the signs of HPC degradation?",
     "Rising T30 and T50 temperatures, rising Ps30, falling phi and falling W31/W32 coolant bleeds, with core speed drifting up."),
    ("What should I do when an engine is in warning status?",
     "Schedule a borescope inspection within 10 cycles, order the HPC blade kit and increase monitoring to every cycle."),
    ("What is the action for a critical engine?",
     "Plan removal at the next maintenance opportunity and do not exceed 15 further cycles without engineering approval."),
    ("What does an anomaly alert mean?",
     "The sensor pattern left the healthy envelope for 3 consecutive cycles; it triggers an engineering review, not necessarily imminent failure."),
    ("How do I tell a sensor fault from real engine degradation?",
     "Compare the channel with physically related channels; a single channel jumping while related channels are steady indicates a sensor fault."),
    ("When is a compressor wash useful?",
     "When degradation is caused by fouling rather than erosion: T30 margin drops but the borescope shows no mechanical damage. It can recover 1-2% HPC efficiency."),
    ("How long must the engine cool before a borescope inspection?", "At least 45 minutes."),
    ("Which HPC stages are inspected with the borescope?", "Stages 3, 5 and 7."),
    ("Why can't raw sensor values be compared across FD002 engines?",
     "FD002 runs across six operating conditions, so values must be compared within the same condition or normalised."),
    ("What fault modes are in FD003?", "HPC degradation and fan degradation."),
    ("How are technicians identified in maintenance logs?",
     "They are pseudonymised with role tokens such as TECH_07 before indexing."),
    ("What damage propagation model did NASA use to simulate degradation in C-MAPSS?",
     "An exponential degradation model of health parameters such as flow and efficiency, with noise, in the C-MAPSS simulation."),
    ("Which engine data rows are quarantined by the pipeline?",
     "Rows with impossible values such as negative temperatures, values above physical limits, or missing channels."),
]


def main() -> None:
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy
    from ragas.run_config import RunConfig

    from thermalguard.rag.llm import get_llm

    rows = []
    for q, ref in GOLDEN:
        msg = ask(q, thread_id=str(uuid.uuid4()))  # fresh thread per question
        rows.append({"user_input": q, "response": msg.content, "reference": ref,
                     "retrieved_contexts": [c["text"] for c in msg.additional_kwargs["citations"]]})
        time.sleep(2)  # stay inside free-tier rate limits

    # reasoning model: give the judge room to finish; Groq allows only n=1 -> strictness=1
    # Separate judge model: avoids self-grading bias and uses its own free-tier daily token quota.
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "qwen/qwen3.8-27b")
    judge = LangchainLLMWrapper(get_llm(max_tokens=8192, model=judge_model))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    result = evaluate(EvaluationDataset.from_list(rows),
                      metrics=[Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference()],
                      llm=judge, embeddings=emb,
                      # free tier: few parallel calls + long timeout, so no job is dropped
                      run_config=RunConfig(max_workers=2, timeout=300, max_retries=10, max_wait=90))
    df = result.to_pandas()
    scores = {c: round(float(df[c].mean()), 3) for c in df.columns if df[c].dtype.kind == "f"}
    scores["scored_rows"] = {c: int(df[c].notna().sum()) for c in df.columns if df[c].dtype.kind == "f"}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "ragas.json").write_text(json.dumps(
        {"n_questions": len(rows), "judge_model": judge_model, "scores": scores,
         "per_question": df.drop(columns=["retrieved_contexts"], errors="ignore").to_dict("records")},
        indent=2, default=str))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
