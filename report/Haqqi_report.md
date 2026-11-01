::: {custom-style="Title page"}
![](report/assets/uowd_logo.jpg){width=2.5cm}

**University of Wollongong in Dubai**

MAAI902 Natural Language Processing and Large Language Models

Assessment 2: Individual Project Report

**Haqqi: Classifying UAE Workers’ Labour Problems and Retrieving the Law in Arabic and English**

Student: Himasri Allu

Student ID: 1789513

Instructor: Dr. Pallavi Ranjan

Session: Autumn 2026

Code: https://github.com/himaallu/Haqqi---NLP-LLM-Project

Colab notebook: https://colab.research.google.com/drive/1CSCJ0YYtAQ1CecSFQpAiW5Jgqw20khWS?usp=sharing

Results produced with commit ef88657
:::

# Abstract

Low-paid migrant employees in the UAE face missing pay, confiscated passports or sudden dismissal but can rarely pay a lawyer to learn which provision of Federal Decree-Law No. 33 of 2021 applies. Haqqi (Arabic for “my right”) takes a worker’s Arabic or English message, assigns a problem category, extracts key facts, finds the matching law articles and drafts a plain-language reply. It was evaluated on 999 genuine posts from Legal Advice Middle East, with 91 test questions annotated blind by the author. For RQ1, a logistic-regression model over BGE-M3 embeddings gave the best category predictions (macro-F1 0.755), compared with 0.662 for Qwen3-8B, 0.490 for Jais 2 8B (both prompted zero-shot) and 0.420 for TF-IDF. For RQ2, hand-written regular expressions were the more precise fact extractor, whereas the LLM recovered more facts but fabricated 16 employer names. For RQ3, a new two-hop search that first gathers ten similar questions already answered by lawyers lifted the proportion of questions with a correct article among the top three from 41% to 95% (exact McNemar p < 0.0001), and the LLM cited correct articles only when these retrieved articles were included in its prompt.

# 1 Introduction

## Background

Most private-sector employees in the UAE come from abroad, and a large share of those working in construction, cleaning and service jobs are poorly paid. The disputes they report most often (salary arrears, end-of-service gratuity, passports held by the employer, recruitment charges and dismissal) fall under Federal Decree-Law No. 33 of 2021, in force since February 2022. Its 74 articles use legal language, and those most affected can seldom afford even a first meeting with a lawyer.

## Gap in Existing Work

Research on legal language processing has expanded, producing benchmarks such as LexGLUE (Chalkidis et al., 2022) and datasets for finding statutory articles that answer questions from ordinary citizens, such as BSARD (Louis & Spanakis, 2022). These resources, however, focus on European languages and on professional users. Nothing equivalent exists for UAE employment law, for Arabic text, or for the casual wording that workers use. Recent evaluations also show that large language models (LLMs) fabricate legal statements (Dahl et al., 2024; Magesh et al., 2024), so an LLM alone cannot be trusted.

## Problem Statement

For any Arabic or English message, the task is to (1) decide which kind of problem it describes, (2) pull out the amount of money, the date, the length of employment and the employer, (3) find the articles of Decree-Law 33/2021 that apply and (4) write a brief reply that cites those articles, with similar quality in both languages.

## Research Questions and Hypotheses

- **RQ1 / H1:** A classifier trained on multilingual BGE-M3 sentence embeddings will reach a higher macro-F1 than both TF-IDF and zero-shot prompting of LLMs, in Arabic and in English.
- **RQ2 / H2:** Hand-written rules will extract facts more precisely, whereas an LLM will recover a larger share of them.
- **RQ3 / H3:** Searching in two hops (from the question to similar answered questions, then to articles) will find a correct article more often than searching directly from the question to the articles. H3b: including the retrieved articles in the prompt will make the LLM cite correct articles more often.

## Contributions

1. A collection of 999 authentic Arabic and English UAE employment questions with lawyer replies, including 452 silver labels and 91 test questions annotated blind by the author.
2. A language-by-language evaluation of TF-IDF, embedding-based and zero-shot LLM classifiers.
3. An assessment of rule-based against LLM-based fact extraction, including how often the LLM fabricates facts.
4. A two-hop retrieval approach in which genuine lawyer replies expand the query, supported by a manually verified keyword list of 925 Arabic and English phrases and a map from categories to articles.
5. A functioning prototype running in Google Colab with an interactive Gradio interface.

# 2 Dataset Selection

Legal Advice Middle East (legaladviceme.com) fits the task closely: it holds real UAE employment questions phrased by workers themselves, in both Arabic and English, and almost every saved question (998 of 999) has a reply from a UAE lawyer. The two-hop method depends on these replies.

Pages were downloaded with Playwright at one request every 12 seconds, with an identifying user agent and respecting robots.txt. Of the 3,414 question links found, 2,065 in the Arabic listing turned out to be translated English posts; these were dropped so that the Arabic portion contains only text originally written in Arabic. With a cap of 1,000 downloads, 999 pages were kept: the most recent 745 English questions, every one of the 252 native Arabic questions and 2 questions mixing both languages. Figure 1 breaks these down by language and year.

The 452 labelled questions combine all Arabic questions with 200 random English ones from 2022 onwards, balancing the languages. One item was discarded because the lawyer’s reply had leaked into the question text. The remaining 451 were divided 80/20, stratified by category with seed 42, giving 360 for training and 91 for testing (48 Arabic, 43 English). The legal corpus is the 74 articles of Decree-Law 33/2021, extracted from the official English (MoHRE) and Arabic (u.ae) PDFs.

**Figure 1**

*Questions by Language and Year*

![](results/figures/fig1_data.png){width=90%}

# 3 Pre-processing Strategy

One routine cleans both stored questions and new messages, so training and live use see identical text.

1. **Language detection:** a message is Arabic when at least 80% of its letters are Arabic, English when at most 20% are, and mixed otherwise; the corpus contained 745 English, 252 Arabic and 2 mixed questions.
2. **Masking of personal data:** regular expressions swap emails, web links, Emirates ID numbers (beginning 784) and UAE phone numbers for placeholder tags such as [PHONE]. Personal names are not detected automatically, so they were reviewed by hand while labelling.
3. **Arabic normalisation:** vowel diacritics and the elongation character (tatweel) are deleted; the hamza forms of alef (أ, إ, آ, ٱ) become a plain alef (ا); alef maksura (ى) becomes yeh (ي); teh marbuta (ة) becomes heh (ه); Eastern Arabic numerals become 0–9; English is lower-cased and extra spaces are removed.
4. **Law-era tag:** anything dated before 2022 is marked “pre2022”, since the lawyer may have answered it under the repealed Federal Law No. 8 of 1980.

BGE-M3 and the LLMs receive masked text; TF-IDF and keyword matching use the normalised version. A morphological analyser such as CAMeL Tools (Obeid et al., 2020) was not needed for this light cleaning and is left for future work.

# 4 Model Selection and Configuration

Every model used is openly available and fits on one NVIDIA A100 GPU in Google Colab. The TF-IDF features combine word unigrams and bigrams with character sequences of two to five letters (with sublinear term frequency), which helps with attached Arabic prefixes. BGE-M3 (Chen et al., 2024), a multilingual encoder of 568M parameters, places Arabic and English text in a shared 1,024-dimensional space and is kept frozen. A logistic-regression layer (C = 5, class weights balanced) sits on top of both feature sets. Qwen3-8B (Yang et al., 2025) was picked as a capable open multilingual LLM and Jais 2 8B Chat as an Arabic-first model developed in the UAE (Sengupta et al., 2023); each runs in bfloat16 with greedy decoding, thinking disabled, and output limits of 20, 120 and 400 tokens for classification, extraction and answering. Retrieval hop 1 takes the ten nearest same-category Q&As by BGE-M3 cosine (near-duplicates excluded); hop 2 keeps the three articles with the highest IDF-weighted keyword scores.

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

An LLM (Claude) assigned silver training labels to the 452 questions by following the author’s written guide, which describes seven categories: unpaid_wages, passport_withheld, gratuity, unfair_termination, recruitment_fees, court_enforcement and other. The author re-annotated the 91 test questions blind to any model output, forming the gold standard. Training touched only the logistic-regression layers; neither BGE-M3 nor the LLMs had their weights updated. LLM replies were cached on Google Drive, so runs can resume and be repeated exactly.

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

RQ1 is scored with macro-F1 (the unweighted mean of per-category F1, so rare categories count equally) and accuracy, overall and per language. RQ2 is scored with precision (correct facts divided by facts returned) and recall (correct facts divided by facts actually present). RQ3 retrieval uses hit@3 (a correct article among the three returned) and recall@3 against the author’s category-to-article map; the 49 “other” questions have no fixed articles, so 42 are scored. Answers are compared with the lawyer reply by BGE-M3 cosine similarity and checked for citations and reply language. Silver–gold agreement is measured with Cohen’s kappa (Cohen, 1960). Confidence intervals at 95% follow the Wilson method, and the paired comparison of one versus two hops uses an exact McNemar test.

## 6.2 Results

**H1 (supported).** The BGE-M3 classifier obtained the best macro-F1 both overall (0.755) and within each language (Table 3, Figure 2), with an accuracy of 0.79 (95% CI 0.70–0.86) compared with 0.68 (0.58–0.77) for TF-IDF. Zero-shot Qwen3-8B ranked second; one unparseable output (of 91) was scored as wrong. Jais 2 did better on Arabic than on English but finished third.

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

**H2 (supported).** For amounts, the regular expressions were right 11 times out of 12 (precision 0.92), while the LLM recovered more of the facts present (recall 0.93 for both amounts and durations) at the cost of more mistakes (Table 4, Figure 3). None of the test questions mentioned an employer by name, but the LLM still returned 16 employer names, every one of them made up.

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

**H3 (supported).** With two hops, 40 of the 42 questions had a correct article in the top three (95%, CI 0.84–0.99); with one hop, only 17 did (41%, CI 0.27–0.56). With at least 23 questions solved only by two hops and at most 2 only by one hop, the exact McNemar test gives p < 0.0001. It held in Arabic (13 to 26 of 27) and English (4 to 14 of 15), recall@3 climbed from 0.16 to 0.48 (Table 5, Figure 4), and on 17 synthetic cases hit@3 went from 0.63 to 1.00.

**Table 5**

*RQ3 Retrieval on 42 Scored Test Questions*

| Method | hit@3 Arabic | hit@3 English | hit@3 All | recall@3 All |
|---|---|---|---|---|
| One hop | 0.48 | 0.27 | 0.41 | 0.16 |
| Two hops | **0.96** | **0.93** | **0.95** | **0.48** |

**Figure 4**

*RQ3 hit@3 and recall@3, One Hop Versus Two Hops*

![](results/figures/fig5_rq3_recall_human.png){width=70%}

**H3b (supported, small sample).** On 30 test questions (15 per language), Qwen3-8B cited article numbers in most answers under every condition, but a mapped article appeared only when the retrieved articles were in the prompt: in 3 of 7 Arabic and 5 of 6 English answers whose category has mapped articles. Similarity to the lawyer reply barely changed (0.65–0.68 Arabic, 0.76–0.78 English), and all answers matched the question’s language (Table 6, Figure 5).

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

Per category, the best classifier did well where the wording is distinctive, such as court_enforcement (F1 0.92) and recruitment_fees (0.89), and worst on unfair_termination (0.60). It assigned unpaid_wages too often (precision 0.58), a category that frequently appears in the same message as a dismissal. BGE-M3 and Qwen3 scored higher in English, while Jais 2 favoured Arabic, suggesting Arabic-focused pre-training helps Arabic at English’s expense. The two-hop improvement reflects a mismatch in vocabulary: workers seldom write legal expressions such as “end-of-service gratuity”, whereas lawyers do, so their answers enrich the query in the manner of relevance-based expansion (Lavrenko & Croft, 2001); the idea resembles HyDE (Gao et al., 2023), except that the added text is authentic rather than generated. Without the retrieved provisions, the LLM confidently states wrong article numbers, as the hallucination literature predicts.

Several biases are possible. Language overlaps with time: 19 of the 48 Arabic test questions were posted before 2022 but none of the English ones were, so some of the Arabic shortfall may stem from the old law rather than from the language itself. The training labels came from an LLM and could tilt models towards LLM-style judgements, even though they agreed closely with the author (88%, kappa 0.82), in line with the annotation quality reported by Gilardi et al. (2023). One person produced the test labels, the keyword table and the article map, and a few categories are tiny (gratuity 1, passport_withheld 2), making their individual scores unreliable. Most synthetic cases (13 of 17) are English.

# 8 Ethical and Societal Implications

The posts are public but personal, so contact details and ID numbers were masked, names were reviewed manually, robots.txt was obeyed and requests were spaced 12 seconds apart. The greatest risk is misleading advice, because the LLM invented facts and article numbers; Haqqi therefore displays the retrieved articles so users can check them, and any public release would need an explicit notice that it does not constitute legal advice. Weaker Arabic results could disadvantage Arabic speakers, so more Arabic data is needed. Domestic workers are governed by a different law, so their cases should be flagged instead of being answered under Decree-Law 33/2021. Socially, Haqqi could give low-paid migrant workers free first guidance in their own language, provided it directs them to the Ministry of Human Resources and Emiratisation (MoHRE) and lawyers rather than replacing them.

**Use of AI tools.** The author conceived the research idea and was responsible for the study design, data collection, test-set annotation, keyword table, article map, experiments and interpretation of results. Claude (Anthropic) served as an assistant for writing and debugging code, explaining methods, preparing the presentation, and drafting and organising this report from the author’s findings; the author reviewed all of this material. LLMs also appear as components of the research itself: Claude generated the silver training labels, and Qwen3-8B and Jais 2 8B were among the models evaluated.

# 9 Limitations and Conclusion

The evaluation rests on a small test set (91 questions, 42 of them for retrieval) annotated by a single person; the article map works at the level of categories rather than individual questions; the RQ1 differences are described with confidence intervals but no formal significance test; and neither stronger retrieval baselines (BM25, dense retrieval over the articles) nor the contribution of Arabic normalisation were measured. Future work should add a second annotator or lawyer, question-level relevance labels, these baselines and more Arabic data.

Overall, the evidence supports all three hypotheses. A frozen multilingual encoder paired with a small trained classifier outperformed 8B LLMs used zero-shot for categorising problems; rule-based extraction remains the safer option when facts must be exact; and drawing on the vocabulary of genuine lawyer answers through two-hop retrieval more than doubled the likelihood of locating the relevant law, which was also the only setting in which the LLM cited correct articles. Combining compact models, retrieval and LLMs can give workers a dependable first step towards their rights.

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
