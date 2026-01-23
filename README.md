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

### Notation
- **Q**: number of queries
- **I**: number of instructions 
- **T_Q**: number of candidate tables per dataset
- **Avg. Row / Avg. Col**: average number of rows / columns per table
- **Hier.**: whether the dataset contains tables with hierarchical headers
- **Relevance Density**
  - **Rel/Q**: average number of relevant tables per query
  - **Comp/I**: average number of instruction-compliant tables per instruction 
- **Len(Q) / Len(I)**: average number of words in queries / instructions
- **C1, C2, S1, S2, S3**: instruction counts by taxonomy categories 

| Dataset  | Q   | I     | T_Q    | Avg. Row | Avg. Col | Hier. | Rel/Q | Comp/I | Len(Q) | Len(I) | C1  | C2  | S1  | S2  | S3  |
|----------|-----|-------|--------|----------|----------|:-----:|------:|-------:|-------:|-------:|----:|----:|----:|----:|----:|
| WQT      | 300 | 1,386 | 23,784 | 8.13     | 4.10     |   —   | 39.22 | 20.79  | 6.14   | 126.53 | 268 | 243 | 297 | 281 | 297 |
| WTR      | 60  | 267   | 9,546  | 5.74     | 22.47    |   —   | 46.55 | 21.22  | 2.80   | 126.21 | 50  | 46  | 59  | 54  | 58  |
| TArX     | 97  | 408   | 11,586 | 10.86    | 5.38     |   ✓   | 30.88 | 16.89  | 5.22   | 43.36  | 69  | 79  | 90  | 83  | 87  |
| IndusTR  | 216 | 928   | 13,258 | 8.15     | 5.33     |   ✓   | 33.81 | 15.38  | 6.95   | 133.55 | 206 | 194 | 211 | 109 | 208 |


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


## Notes


### 📌 Dataset Type to Paper Dimension Mapping

The following table provides the cross-reference between the `type` field used in this dataset and the evaluation dimensions defined in our paper:

| Dataset Type | Paper Dimension ID | Dimension Name |
| --- | --- | --- |
| **type 1.1** | **C1** | Semantic Boundary Constraint |
| **type 1.2** | **C2** | Exclusive Topic Constraint |
| **type 2.1** | **S1** | Attribute-centric Structural Constraint |
| **type 2.2** | **S2** | Entity-centric Structural Constraint |
| **type 2.3.*** | **S3** | Granularity-centric Structural Constraint |


