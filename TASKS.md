# gh-issues-triage backlog

The shared board for every session. Statuses: Ready -> In progress -> Review -> Done.
A ticket owns the files/functions it lists; two tickets in progress must never own the same thing.

Template:

    ### T-<n>: <title>
    Status: Ready | In progress (branch) | Review (branch) | Done
    Owns: <files, functions or sections>
    Blocked by: <ticket or "none">
    Why: <one or two lines>
    Done when:
    - [ ] <something a reviewer can check without asking anyone>

## Ready

### T-1: Decide on issue text: store or reprocess bodies?
Status: Ready
Owns: decision only, then a follow-up ticket
Blocked by: none
Why: Accuracy is stuck near the logistic baseline (replayed ROC AUC ~0.66) with counts and title flags only. The next lever is the issue text, which needs a privacy decision.
Done when:
- [ ] The user has picked an option on the Command Center ask
- [ ] If yes: a follow-up ticket lists the storage change and the replay comparison to run

### T-2: Predict time to first maintainer response as a second label
Status: Ready
Owns: src/ghtriage/features.py (label only); config; tests
Blocked by: none
Why: Closed-for-any-reason mixes fixes with duplicates; first response is closer to what maintainers act on.
Done when:
- [ ] Label defined without post-creation leakage
- [ ] Replayed weekly ROC AUC reported for both labels
- [ ] README says which label the site uses and why

## In progress

## Review

## Done
