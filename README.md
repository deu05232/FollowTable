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

## Benchmark Statistics

The table below summarizes key statistics of **FollowTable** across its constituent datasets.  
Let \(|\mathcal{Q}|\), \(|\mathcal{I}|\), and \(|\mathcal{T}_{\mathcal{Q}}|\) denote the number of **queries**, **instructions**, and **candidate tables** per dataset, respectively.

- **Avg. Row / Avg. Col**: average number of rows / columns per table  
- **Hier.**: whether the dataset contains tables with *hierarchical headers*  
- **Relevance Density**:
  - \(|\mathcal{T}^+_{\mathcal{Q}}|/|\mathcal{Q}|\): average # relevant tables per query  
  - \(|\mathcal{T}^+_{\mathcal{Q},\mathcal{I}}|/|\mathcal{I}|\): average # instruction-compliant tables per instruction  
- **Len (Q) / Len (I)**: average number of words in queries / instructions  
- **Instruction Scale by Type (C1, C2, S1, S2, S3)**: distribution of instructions across taxonomy categories (as defined in our paper)

| Dataset | \|\u211A\| (Queries) | \|\u2130\| (Instructions) | \|\u2131_Q\| (Candidate Tables) | Avg. Row | Avg. Col | Hier. | \|\u2131⁺_Q\|/\|\u211A\| | \|\u2131⁺_{Q,I}\|/\|\u2130\| | Len (Q) | Len (I) | C1 | C2 | S1 | S2 | S3 |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WQT | 300 | 1,386 | 23,784 | 8.13 | 4.10 | — | 39.22 | 20.79 | 6.14 | 126.53 | 268 | 243 | 297 | 281 | 297 |
| WTR | 60 | 267 | 9,546 | 5.74 | 22.47 | — | 46.55 | 21.22 | 2.80 | 126.21 | 50 | 46 | 59 | 54 | 58 |
| TArX | 97 | 408 | 11,586 | 10.86 | 5.38 | ✓ | 30.88 | 16.89 | 5.22 | 43.36 | 69 | 79 | 90 | 83 | 87 |
| IndusTR | 216 | 928 | 13,258 | 8.15 | 5.33 | ✓ | 33.81 | 15.38 | 6.95 | 133.55 | 206 | 194 | 211 | 109 | 208 |

**Notes**
- “Hier.” indicates whether the dataset includes tables with hierarchical headers.
- The taxonomy category counts (C1, C2, S1, S2, S3) correspond to Columns 1.1–2.3 in our instruction taxonomy; see the paper for definitions.

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
