Goal: Reproducible headless Go2Trot intervention experiments with interchangeable artifacts and models.
Status: Phase 1 — goal recorded; static dependency review complete, runtime compatibility gate open — 2026-11-15.
Open questions: Initial CBF/MPPI model and objective combinations; dependency compatibility and selection.

## 1. Goal

### 1.a. Goal statements
- Provide a reusable intervention library and reproducible headless evaluation script for Go2Trot joint-target actions.
- Support working max_Q, nearest-feasible Q threshold, sequential-rejection Q threshold, CBF, and MPPI methods.
- Allow experiments with user-selected policies, Q artifacts, dynamics models, barriers, and objectives.

### 1.b. Success criteria
- Every intervention method runs end-to-end with at least one compatible, documented artifact/model combination.
- Controlled comparisons report task performance, constraint violations, intervention frequency/magnitude, and computation cost.
- Experiments record inputs, settings, and randomness sufficiently for repeatable comparisons, with numerical nondeterminism disclosed.

### 1.c. Constraints
- Consume existing policies and Q artifacts; training them is outside scope.
- Permit alternative model fidelities without presuming which fidelity/performance tradeoff works best.
- Reject incompatible inputs and expose failures; empirical improvements are not safety guarantees.

### 1.d. Questions and status
- **Answered:** Initial scope is headless Go2Trot joint-target intervention, with all methods operational.
- **Answered:** Model and objective choices remain experimental; concrete implementations are required, not interfaces alone.
- **Open:** Which initial CBF/MPPI model and objective combinations and compatible dependency versions will be supported?

All methods working requires at least one concrete CBF barrier/model and MPPI dynamics/cost combination during architecture planning. Loading arbitrary Q artifacts does not supply these automatically.

The user requested dependency compatibility research before proceeding beyond this phase. No architecture or dependency selection is approved yet.

Dependency evidence: [compatibility research](intervention_testing_compatibility.md). This review found explicit incompatible installation choices; it does not establish successful resolution, installation, or execution.
