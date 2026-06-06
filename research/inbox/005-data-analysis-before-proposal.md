---

[English](005-data-analysis-before-proposal.md) | [Chinese](005-data-analysis-before-proposal.zh.md)
from: human
date: "2026-04-22"
priority: urgent
subject: "Data analysis must be done before proposing an experiment"
---

Page-wise NTP proposal for outbox/002:

**Question**: You did not analyze the data before proposing PW-NTP. The core assumption of PW-NTP is "there are multiple positive interactions within the same exposure page", but you don't know what proportion of this situation accounts for our data. If most sessions only have 1 click per session, PW-NTP will make little difference.

**Require**:

1. **Data analysis must be done before making any experimental proposal**. Write analysis scripts (put in `experiments/scripts/`) and use actual data to verify whether the hypothesis is true. For example:
   - Count the average number of positive interactions within each session/page
   - The distribution of positive interactions (1, 2, 3...what proportion each accounts for)
   - Proportion of multi-positive interactive sessions

2. **Precipitate analysis tools** so that these scripts can be reused for a long time. Comply with program.md §10 Proactive Tooling.

3. After the analysis is completed, attach the results to the proposal. If the data doesn't support the hypothesis, don't mention it.

4. **This rule applies to all future proposals**, not just PW-NTPs. Every proposal must be supported by data.

Regarding source code modification authorization: Complete the data analysis first, and then talk if the data supports it.

Regarding the session segmentation standard: This should also be determined through data analysis - count the session distribution under different time intervals (10min/30min/1h/1d) and choose the most reasonable threshold. Don't ask me, analyze the data.
