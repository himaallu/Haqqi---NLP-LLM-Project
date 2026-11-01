::: {custom-style="Title page"}
**University of Wollongong in Dubai**

MAAI902 Natural Language Processing and Large Language Models

Assessment 2: Individual Project Report

**Haqqi: Classifying UAE Workers’ Labour Problems and Retrieving the Law in Arabic and English**

Student: Himasri Allu

Student ID: 1789513

Instructor: [Instructor name]

Session: Autumn 2026

Date: 1 November 2026

Code: https://github.com/himaallu/Haqqi---NLP-LLM-Project

Colab notebook: https://colab.research.google.com/drive/1CSCJ0YYtAQ1CecSFQpAiW5Jgqw20khWS?usp=sharing

Results produced with commit ef88657
:::

# Abstract

Low-paid migrant workers in the United Arab Emirates (UAE) often face unpaid wages, withheld passports or unfair dismissal, but cannot afford a lawyer to tell them which part of Federal Decree-Law No. 33 of 2021 protects them. This project builds and evaluates Haqqi (“my right”), a pipeline that reads a worker’s message in Arabic or English, classifies the problem, extracts key facts, retrieves the relevant law articles and drafts a plain-language answer. Using 999 real questions from Legal Advice Middle East and a blind, author-labelled test set of 91 questions, three research questions were tested. (RQ1) BGE-M3 embeddings with logistic regression classified problem types best (macro-F1 0.755), ahead of zero-shot Qwen3-8B (0.662), Jais 2 8B (0.490) and TF-IDF (0.420). (RQ2) Regex rules extracted facts with higher precision, while an LLM achieved higher recall but invented 16 employer names. (RQ3) A novel two-hop retrieval, which first finds ten similar lawyer-answered questions, raised the share of questions with a correct article in the top three from 41% to 95% (exact McNemar p < 0.0001). Supplying these articles to the LLM was the only condition in which its answers cited correct articles.

# 1 Introduction

## Background

The UAE’s private-sector workforce is largely migrant, and many workers in construction, cleaning and services earn low wages. Their most common problems (unpaid salary, end-of-service gratuity, passport retention, recruitment fees and dismissal) are regulated by Federal Decree-Law No. 33 of 2021, which came into force in February 2022. The law has 74 articles written for legal professionals, and a first consultation with a lawyer is often unaffordable for the workers who need it most.

## Gap in Existing Work

Legal NLP has grown quickly, with benchmarks such as LexGLUE (Chalkidis et al., 2022) and statutory article retrieval for lay questions such as BSARD (Louis & Spanakis, 2022). However, this work targets European languages and legal professionals. There is no comparable resource or evaluation for UAE labour law, for Arabic, or for the informal way workers describe their problems. Recent studies also show that large language models (LLMs) hallucinate legal content (Dahl et al., 2024; Magesh et al., 2024), so an LLM alone is not a safe answer.

## Problem Statement

Given a free-text message in Arabic or English, the system must (1) identify the problem type, (2) extract the amount, date, duration and employer, (3) retrieve the most relevant articles of Decree-Law 33/2021 and (4) produce a short answer that cites them, while working equally well in both languages.

## Research Questions and Hypotheses

- **RQ1 / H1:** Multilingual sentence embeddings (BGE-M3) with a trained classifier will achieve higher macro-F1 than TF-IDF and than zero-shot LLM prompting, in both languages.
- **RQ2 / H2:** Rule-based extraction will be more precise, and LLM extraction will have higher recall.
- **RQ3 / H3:** Two-hop retrieval (question → similar lawyer-answered questions → articles) will retrieve correct articles more often than one-hop retrieval (question → articles). H3b: giving the retrieved articles to the LLM will increase correct article citations in its answers.

## Contributions

1. A bilingual dataset of 999 real UAE employment questions with lawyer answers, 452 silver labels and a 91-question test set labelled blind by the author.
2. A per-language comparison of TF-IDF, embedding and zero-shot LLM classifiers.
3. A comparison of rule-based and LLM fact extraction that quantifies hallucination.
4. A two-hop retrieval method that uses real lawyer answers as query expansion, with a hand-checked bilingual keyword table (925 phrases) and a problem-type-to-article map.
5. A working prototype (Google Colab with a Gradio demo).

# 2 Dataset Selection

Legal Advice Middle East (legaladviceme.com) was selected because it contains what the task requires: real workers describing UAE employment problems in their own words, in Arabic and English, with answers from UAE lawyers (998 of 999 saved questions have at least one answer). These answers make two-hop retrieval possible.

Collection was done with Playwright, with a 12-second delay between requests, an identifying user agent, and no URLs disallowed by robots.txt. From 3,414 question links, 2,065 links in the Arabic listing were translations of English posts and were excluded, so the Arabic data is native Arabic writing. Within a 1,000-page download limit, 999 pages were saved: the 745 newest English questions, all 252 native Arabic questions and 2 mixed-language questions. Figure 1 shows the language and year distribution.

For labelling, all 252 Arabic questions and 200 randomly selected English questions from 2022 or later were chosen (452), which keeps the two languages balanced. One question whose text contained the lawyer’s answer was removed, leaving 451, which were split 80/20 (stratified by type, seed 42) into 360 training and 91 test questions (48 Arabic, 43 English). The law corpus is the 74 articles of Decree-Law 33/2021, split from the official English (MoHRE) and Arabic (u.ae) PDFs.

**Figure 1**

*Questions by Language and Year*

![](results/figures/fig1_data.png){width=90%}

# 3 Pre-processing Strategy

One function prepares both the stored data and live messages, so training and deployment see identical text.

1. **Language detection:** the share of Arabic letters among all letters (≥ 0.8 Arabic, ≤ 0.2 English, otherwise mixed); this gave 745 English, 252 Arabic and 2 mixed questions.
2. **Masking of personal data** with regular expressions: emails, URLs, Emirates ID numbers (784-…) and UAE phone numbers are replaced with tags such as [PHONE]. Names are not masked automatically and were checked manually during labelling.
3. **Arabic normalisation:** diacritics and tatweel removed; the alef variants (أ, إ, آ, ٱ) mapped to bare alef (ا); alef maksura (ى) mapped to yeh (ي); teh marbuta (ة) mapped to heh (ه); Arabic-Indic digits converted to 0–9; English lower-cased; whitespace collapsed.
4. **Law-era tag:** questions dated before 2022 are tagged “pre2022”, because they may have been answered under the old Federal Law No. 8 of 1980.

Masked text is the input to BGE-M3 and the LLMs, which handle Arabic themselves; normalised text is used for TF-IDF and keyword matching. Morphological tools such as CAMeL Tools (Obeid et al., 2020) were not needed for this light normalisation and are left for future work.

# 4 Model Selection and Configuration

All models are open and run on a single NVIDIA A100 in Google Colab. TF-IDF uses word 1–2-grams and character 2–5-grams (sublinear term frequency), which cope with Arabic prefixes. BGE-M3 (Chen et al., 2024) is a 568M-parameter multilingual encoder that maps Arabic and English into one 1,024-dimensional space; it is used frozen. Both classifiers use logistic regression (C = 5, balanced class weights). Qwen3-8B (Yang et al., 2025) was chosen as a strong open multilingual LLM, and Jais 2 8B Chat as an Arabic-centric model built in the UAE (Sengupta et al., 2023); both run in bfloat16 with greedy decoding, thinking mode off, and at most 20, 120 or 400 new tokens for classification, extraction and answers. Retrieval hop 1 uses BGE-M3 cosine similarity (k = 10, same predicted type, near-duplicates removed); hop 2 uses IDF-weighted keyword matching (top 3).

**Table 1**

*Alternatives Considered*

| Component | Chosen | Alternatives | Decision |
|---|---|---|---|
| Problem-type classifier | BGE-M3 + LR | TF-IDF + LR; zero-shot Qwen3-8B; zero-shot Jais 2 8B | All tested (RQ1); BGE-M3 best |
| LLM adaptation | Zero-shot prompting | Fine-tuning an 8B LLM | Not run: 360 examples too few, high GPU cost |
| Arabic pre-processing | Light rule-based normalisation | CAMeL Tools segmentation and lemmatisation | Not needed for current methods; future work |
| Fact extraction | Regex rules (for facts shown) | LLM extraction (Qwen3-8B) | Both tested (RQ2) |
| Article retrieval | Two hops via lawyer Q&As | One hop (question only); BM25; dense retrieval over articles | One vs two hops tested (RQ3); others future work |
| Answer generation | Qwen3-8B with Q&As and articles | Question only; question + Q&As | All three tested (RQ3) |

# 5 Training / Fine-tuning Process

Silver training labels for 452 questions were produced by an LLM (Claude) following the author’s written labelling guide, which defines seven types: unpaid_wages, passport_withheld, gratuity, unfair_termination, recruitment_fees, court_enforcement and other. The 91 test questions were then relabelled blind by the author, without seeing any model suggestion, to serve as the gold standard. Only the logistic-regression heads were trained; BGE-M3 and the LLMs were not fine-tuned. LLM outputs were cached on Google Drive so that runs could resume after a disconnect and be reproduced exactly.

**Table 2**

*Experimental Runs*

| Run | Purpose | Setting | Outcome |
|---|---|---|---|
| 1 | Data collection | Local, Playwright, 12 s delay | 3,414 links; 999 pages |
| 2 | Silver labelling | LLM following the guide | 452 labelled questions |
| 3 | Blind gold labelling | Author, spreadsheet | 91 test labels; kappa 0.82 with silver |
| 4 | Keyword table and article map | Author | 925 keywords, 74 articles checked; map for 6 types |
| 5 | Classical models and retrieval | CPU | TF-IDF, BGE-M3 + LR, one- and two-hop retrieval |
| 6 | Full pipeline, first run | Colab A100 | One Qwen3 reply unreadable, so Qwen3 missing from RQ1; citation metric counted “other” as failures |
| 7 | Final run after fixes | Colab A100 | Unreadable reply counted as wrong; citations scored only for mapped types; final results |

# 6 Evaluation Metrics and Results

## 6.1 Metrics and Statistical Analysis

RQ1 uses macro-F1 (the F1 averaged over the seven types, so rare types count equally) and accuracy, overall and per language. RQ2 uses precision (correct ÷ extracted) and recall (correct ÷ present) per fact. RQ3 retrieval uses hit@3 (at least one correct article in the top three) and recall@3, judged against the author’s problem-type-to-article map; the 49 test questions of type “other” have no fixed articles, leaving 42 scored questions. Answers are measured by BGE-M3 cosine similarity to the real lawyer answer, the share citing any article, the share citing a mapped article, and language match. Agreement between silver and gold labels uses Cohen’s kappa (Cohen, 1960). Uncertainty is reported as 95% Wilson intervals; the paired one- vs two-hop comparison uses an exact McNemar test.

## 6.2 Results

**H1 (supported).** BGE-M3 + LR achieved the highest macro-F1 overall (0.755) and in both languages (Table 3, Figure 2). Its accuracy was 0.79 (95% CI 0.70–0.86), against 0.68 (0.58–0.77) for TF-IDF. Zero-shot Qwen3-8B was second; one of its 91 replies was unreadable after three attempts and was counted as wrong. Jais 2 performed better in Arabic than English but was third overall.

**Table 3**

*RQ1 Macro-F1 (Accuracy) on 91 Test Questions*

| Model | All | Arabic (48) | English (43) |
|---|---|---|---|
| BGE-M3 + LR | **0.755** (0.791) | **0.704** | **0.793** |
| Qwen3-8B, zero-shot | 0.662 (0.725) | 0.573 | 0.736 |
| Jais 2 8B Chat, zero-shot | 0.490 (0.429) | 0.547 | 0.398 |
| TF-IDF + LR | 0.420 (0.681) | 0.402 | 0.256 |

**Figure 2**

*RQ1 Macro-F1 by Model and Language*

![](results/figures/fig3_rq1_human.png){width=85%}

**H2 (supported).** Rules were more precise for amounts (11 of 12 correct, precision 0.92), while the LLM found more facts (recall 0.93 for amounts and durations) but made more errors (Table 4, Figure 3). No test question named an employer, yet the LLM produced 16 employer names, all invented.

**Table 4**

*RQ2 Fact Extraction: Precision / Recall (Gold Count)*

| Fact | Gold | Rules | LLM (Qwen3-8B) |
|---|---|---|---|
| Amount | 14 | **0.92** / 0.79 | 0.65 / **0.93** |
| Date | 13 | 0.57 / 0.31 | **0.64** / **0.54** |
| Duration | 14 | 0.47 / 0.64 | **0.54** / **0.93** |
| Employer | 0 | no output | 16 invented |

**Figure 3**

*RQ2 Precision and Recall per Fact*

![](results/figures/fig4_rq2_human.png){width=90%}

**H3 (supported).** Two-hop retrieval found a correct article in the top three for 40 of 42 questions (95%, CI 0.84–0.99), against 17 of 42 (41%, CI 0.27–0.56) for one hop. At least 23 questions were hits only with two hops and at most 2 only with one hop, so the exact McNemar test gives p < 0.0001 even in the least favourable case. The gain held in Arabic (13 → 26 of 27) and English (4 → 14 of 15), recall@3 rose from 0.16 to 0.48 (Table 5, Figure 4), and on 17 synthetic cases hit@3 rose from 0.63 to 1.00.

**Table 5**

*RQ3 Retrieval on 42 Scored Test Questions*

| Method | hit@3 Arabic | hit@3 English | hit@3 All | recall@3 All |
|---|---|---|---|---|
| One hop | 0.48 | 0.27 | 0.41 | 0.16 |
| Two hops | **0.96** | **0.93** | **0.95** | **0.48** |

**Figure 4**

*RQ3 hit@3 and recall@3, One Hop Versus Two Hops*

![](results/figures/fig5_rq3_recall_human.png){width=70%}

**H3b (supported, small sample).** On 30 test questions (15 per language), Qwen3-8B cited article numbers in most answers in every condition, but cited a mapped article only when the retrieved articles were supplied: 3 of 7 Arabic and 5 of 6 English answers with mapped types. Similarity to the lawyer answer changed little (0.65–0.68 Arabic, 0.76–0.78 English), and all answers were in the question’s language (Table 6, Figure 5).

**Table 6**

*RQ3 Answers by Condition*

| Condition | Similarity AR / EN | Cites any article AR / EN | Cites mapped article AR / EN |
|---|---|---|---|
| Question only | 0.649 / 0.765 | 0.73 / 0.80 | 0.00 / 0.00 |
| + 10 Q&As | **0.678** / **0.782** | 0.60 / 0.67 | 0.00 / 0.00 |
| + Q&As + 3 articles | 0.657 / 0.764 | 0.73 / 0.93 | **0.43** / **0.83** |

**Figure 5**

*Answer Similarity to the Lawyer Answer by Condition*

![](results/figures/fig6_rq3_answers_human.png){width=80%}

# 7 Analysis of Performance and Potential Biases

Per class, the best classifier was strongest where vocabulary is specific, for example court_enforcement (F1 0.92) and recruitment_fees (0.89), and weakest on unfair_termination (0.60). It over-predicted unpaid_wages (precision 0.58), which often co-occurs with dismissal in the same message. English scored higher than Arabic for BGE-M3 and Qwen3, while Jais 2 reversed this pattern, suggesting that Arabic-centric pre-training helps Arabic but not English. The two-hop gain shows a vocabulary gap: workers rarely use legal terms such as “end-of-service gratuity”, but lawyer answers do, so they act as query expansion (Lavrenko & Croft, 2001), similar in spirit to HyDE (Gao et al., 2023) but with real rather than generated text. The answer results agree with work on legal hallucination: without retrieved law, the LLM confidently cites incorrect article numbers.

Several potential biases apply. Language and time are confounded: 19 of the 48 Arabic test questions predate 2022, while none of the English ones do, so part of the Arabic gap may reflect the old law rather than the language. Training labels were produced by an LLM, which may favour LLM-like decisions, although agreement with the author was high (88%, kappa 0.82; Gilardi et al., 2023 report similar annotation quality). A single annotator wrote the test labels, keyword table and article map, and some classes are very small (gratuity 1, passport_withheld 2), so per-class scores for them are unreliable. The synthetic cases are mostly English (13 of 17).

# 8 Ethical and Societal Implications

The questions are public, but they contain personal stories, so emails, phone and ID numbers were masked and names were checked manually; collection respected robots.txt and used a 12-second delay. The main risk is incorrect advice: the LLM hallucinated facts and article numbers, so Haqqi shows the retrieved articles for the user to verify, and a deployed version needs a clear statement that it is not legal advice. Lower Arabic performance could disadvantage Arabic speakers, which calls for more Arabic data. Domestic workers are covered by a separate law and should be flagged rather than answered under Decree-Law 33/2021. For society, Haqqi could give low-paid migrant workers a first, free orientation in their own language, but it must direct them to the Ministry of Human Resources and Emiratisation (MoHRE) and to lawyers rather than replace them.

**Use of AI tools.** The research idea, study design, data collection, labelling of the test set, keyword table, article map, experiments and interpretation are the author’s own work. Claude (Anthropic) was used as an assistant to help write and debug code, explain methods, prepare the presentation and draft and structure this report from the author’s results; all content was reviewed by the author. An LLM was also used as a research component: Claude produced the silver training labels, and Qwen3-8B and Jais 2 8B were evaluated as models in the study.

# 9 Limitations and Conclusion

The test set is small (91 questions, 42 for retrieval), labelled by one person; the article map is defined per problem type rather than per question; no significance test was run for the RQ1 differences beyond confidence intervals; and stronger retrieval baselines (BM25, dense retrieval over articles) and the effect of Arabic normalisation were not tested. Future work should add a second annotator or a lawyer, per-question relevance labels, these baselines, morphological Arabic processing and more Arabic data.

In conclusion, all three hypotheses were supported. A frozen multilingual encoder with a small trained classifier outperformed 8B zero-shot LLMs for problem classification; rules remain the safer choice for precise facts; and borrowing the vocabulary of real lawyer answers through two-hop retrieval more than doubled the chance of finding the right law, which in turn was the only way the LLM cited correct articles. Haqqi shows that a careful combination of small models, retrieval and LLMs can give workers a reliable first step towards their rights.

# References

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., … Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems, 33*, 1877–1901.

Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., & Aletras, N. (2022). LexGLUE: A benchmark dataset for legal language understanding in English. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (pp. 4310–4330). Association for Computational Linguistics.

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In *Findings of the Association for Computational Linguistics: ACL 2024*. Association for Computational Linguistics.

Chiticariu, L., Li, Y., & Reiss, F. R. (2013). Rule-based information extraction is dead! Long live rule-based information extraction systems! In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing* (pp. 827–832). Association for Computational Linguistics.

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46.

Dahl, M., Magesh, V., Suzgun, M., & Ho, D. E. (2024). Large legal fictions: Profiling legal hallucinations in large language models. *Journal of Legal Analysis, 16*(1), 64–93.

Gao, L., Ma, X., Lin, J., & Callan, J. (2023). Precise zero-shot dense retrieval without relevance labels. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics* (pp. 1762–1777). Association for Computational Linguistics.

Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *Proceedings of the National Academy of Sciences, 120*(30), e2305016120.

Lavrenko, V., & Croft, W. B. (2001). Relevance-based language models. In *Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 120–127). ACM.

Legal Advice Middle East. (n.d.). *Employment questions*. Retrieved October 2026, from https://legaladviceme.com/questions

Louis, A., & Spanakis, G. (2022). A statutory article retrieval dataset in French. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (pp. 6789–6803). Association for Computational Linguistics.

Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2024). *Hallucination-free? Assessing the reliability of leading AI legal research tools* (arXiv:2405.20362). arXiv.

Obeid, O., Zalmout, N., Khalifa, S., Taji, D., Oudah, M., Alhafni, B., … Habash, N. (2020). CAMeL Tools: An open source Python toolkit for Arabic natural language processing. In *Proceedings of the 12th Language Resources and Evaluation Conference* (pp. 7022–7032). European Language Resources Association.

Sengupta, N., Sahu, S. K., Jia, B., Katipomu, S., Li, H., Koto, F., … Xing, E. (2023). *Jais and Jais-chat: Arabic-centric foundation and instruction-tuned open generative large language models* (arXiv:2308.16149). arXiv.

United Arab Emirates. (2021). *Federal Decree-Law No. 33 of 2021 on the regulation of labour relations*. Ministry of Human Resources and Emiratisation.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., … Qiu, Z. (2025). *Qwen3 technical report* (arXiv:2505.09388). arXiv.

# Appendices

## Appendix A: Problem Types (Labelling Guide Summary)

| Type | Definition |
|------------------------|------------------------------------------------------------------------|
| unpaid_wages | Salary not paid, paid late or only in part (including overtime and allowances) |
| passport_withheld | Employer keeps the passport, Emirates ID or other official documents |
| gratuity | End-of-service gratuity not paid, or how much is owed |
| unfair_termination | Dismissed without valid reason or notice, or compensation for dismissal |
| recruitment_fees | Worker charged recruitment, visa or hiring costs |
| court_enforcement | Existing case, judgment or MoHRE decision: enforcement, appeals, fees, hearings |
| other | Anything else (topic noted, e.g. annual leave, visa ban) |

## Appendix B: Problem-Type-to-Article Map (RQ3 Answer Key)

| Type | Articles of Decree-Law 33/2021 |
|------------------------|------------------------------------------------------------------------|
| unpaid_wages | 19, 22, 25, 26, 53 |
| passport_withheld | 13 |
| gratuity | 51, 52, 53 |
| unfair_termination | 9, 42, 43, 44, 47 |
| recruitment_fees | 6, 9 |
| court_enforcement | 54, 55, 65 |
| other | not scored |

## Appendix C: Per-Class Results of the Best Classifier (BGE-M3 + LR)

| Type | Precision | Recall | F1 | Support |
|------------------------------|-------------|-------------|-------------|-------------|
| unpaid_wages | 0.58 | 1.00 | 0.74 | 7 |
| passport_withheld | 1.00 | 1.00 | 1.00 | 2 |
| gratuity | 0.20 | 1.00 | 0.33 | 1 |
| unfair_termination | 0.55 | 0.67 | 0.60 | 9 |
| recruitment_fees | 0.80 | 1.00 | 0.89 | 4 |
| court_enforcement | 0.94 | 0.89 | 0.92 | 19 |
| other | 0.92 | 0.71 | 0.80 | 49 |

## Appendix D: Reproducibility

The repository contains the scraper, pre-processing, labelling tools, keyword table, article map and the Colab notebook (`notebooks/Haqqi_MAAI902.ipynb`). To reproduce: upload `haqqi_drive_data.zip` to Google Drive as `MyDrive/haqqi/`, open the notebook on an A100, add a Hugging Face token as the secret `HF_TOKEN` and select *Run all*. All tables are written to `results/` and summarised in `results/final_results.md`; LLM outputs are cached in `results/cache/`.
