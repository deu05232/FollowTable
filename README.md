# FollowTable_Benchmark

Traditional Table Retrieval (TR) is commonly treated as an ad-hoc retrieval problem, where relevance is largely defined by topical semantic similarity. However, with the rapid adoption of LLM-based agentic systems, structured data access is increasingly **instruction-driven**: users (or agents) often specify explicit constraints over **table content** and **schema**, and relevance depends on satisfying those constraints—not just matching a topic.

This repository introduces **Instruction-Following Table Retrieval (IFTR)**, a new retrieval setting where models must **jointly** satisfy:
- **Topical relevance** (semantic match to the query intent), and
- **Fine-grained instruction constraints** (explicit inclusion/exclusion rules and schema-specific requirements).

## What is IFTR?

In IFTR, a request may include constraints such as:
- **Content scope constraints**: *include X*, *exclude Y*, *only tables mentioning...*, *avoid tables containing...*
- **Schema-grounded constraints**: requirements tied to **columns**, **field semantics**, or **representation granularity** (e.g., needing a specific column like `country`, preferring entity-level vs. aggregated statistics, etc.)

These capabilities are largely missing in many existing retrievers, which often over-rely on surface-level semantic similarity.

## Key Challenges

We identify two core challenges for IFTR:

1. **Sensitivity to content scope**
   - Correctly honoring inclusion/exclusion constraints and other scope restrictions.

2. **Awareness of schema-grounded requirements**
   - Understanding column semantics and selecting tables that match the requested representation granularity.

## FollowTable Benchmark

We introduce **FollowTable**, the first large-scale benchmark designed specifically for IFTR. It is built using a **taxonomy-driven annotation pipeline** to ensure systematic coverage of instruction types and constraint patterns.

## Evaluation: Instruction Responsiveness Score

To evaluate whether a retriever *actually follows instructions*, we propose a new metric:

- **Instruction Responsiveness Score (IRS)**: measures whether retrieval rankings **consistently adapt to user instructions** compared to a **topic-only baseline**.

This helps distinguish models that merely retrieve topically related tables from those that reliably incorporate instruction constraints.

## Findings

Our experiments show that existing retrieval models struggle with fine-grained instruction following over tabular data. Common failure modes include:
- Systematic bias toward **surface-level semantic cues**
- Limited ability to satisfy **schema-grounded constraints**

These results highlight substantial room for improvement in instruction-following retrieval for structured data.

## Release

The **FollowTable benchmark is publicly released** as part of this project.
