---------------------------- MODULE ExperimentLifecycle ----------------------------
(***************************************************************************)
(* Abstract lifecycle model of one EdgeFault-Bench experiment.             *)
(*                                                                         *)
(* The states, transition guards, and invariants mirror                    *)
(* docs/formal-spec-draft.md (sections 3-4); section 7 of that document    *)
(* maps every invariant to the Python mechanism that enforces it and to    *)
(* the mutation that must be rejected. This module is the model-checked    *)
(* layer of that stack, not a proof about the Python implementation.       *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Recordings,        \* abstract recording identifiers
          Windows,           \* abstract window identifiers
          Seeds              \* the task's registered seeds

Partitions == {1, 2, 3}
PTrain == 1
PVal == 2
PTest == 3
TrainingSide == {PTrain, PVal}

VARIABLES
  phase,          \* lifecycle state
  sourceOk,       \* INV-1: registered sizes and SHA-256 matched at acquire
  part,           \* INV-2: recording -> partition assignment
  recOf,          \* INV-2: window -> owning recording
  winPart,        \* INV-2: partition claimed by each window's metadata
  fitScope,       \* INV-3: partitions whose data reached preprocessing fit
  pluginOk,       \* INV-6: external plugin satisfied its contract
  trained,        \* INV-4: seeds that completed training
  evaluated,      \* INV-3/4: seeds whose test evaluation ran exactly once
  bundleOk,       \* INV-5: result bundle complete and hash-consistent
  rejectReason    \* diagnostic recorded on the fail-closed path

vars == <<phase, sourceOk, part, recOf, winPart, fitScope, pluginOk,
          trained, evaluated, bundleOk, rejectReason>>

\* ------------------------------------------------------------------ guards

PartitionValid(p) ==
  /\ p \in [Recordings -> Partitions]
  /\ \A pt \in Partitions : \E r \in Recordings : p[r] = pt

WindowInheritance(p, own, wp) == \A w \in Windows : wp[w] = p[own[w]]

FitScopeValid(fs) == fs \subseteq TrainingSide

Committed ==
  phase \in {"TASK_AUDITED", "DATA_PREPARED", "MODEL_CONSTRUCTED",
             "RESULT_VALIDATED", "PUBLISHED"}

\* ------------------------------------------------------------------- init

Init ==
  /\ phase = "EMPTY"
  /\ sourceOk = FALSE
  /\ part = [r \in Recordings |-> PTrain]
  /\ recOf = [w \in Windows |-> CHOOSE r \in Recordings : TRUE]
  /\ winPart = [w \in Windows |-> PTrain]
  /\ fitScope = {}
  /\ pluginOk = FALSE
  /\ trained = {}
  /\ evaluated = {}
  /\ bundleOk = FALSE
  /\ rejectReason = "none"

\* -------------------------------------------------------------- transitions

\* t1: checksum-verified acquisition (guard: INV-1)
Acquire ==
  /\ phase = "EMPTY"
  /\ \E ok \in BOOLEAN :
       /\ sourceOk' = ok
       /\ IF ok
          THEN /\ phase' = "SOURCES_VERIFIED" /\ rejectReason' = "none"
          ELSE /\ phase' = "REJECTED" /\ rejectReason' = "source-integrity"
  /\ UNCHANGED <<part, recOf, winPart, fitScope, pluginOk,
                 trained, evaluated, bundleOk>>

\* t2: pre-training leakage audit (guard: INV-2, recording level)
Audit ==
  /\ phase = "SOURCES_VERIFIED"
  /\ part' \in [Recordings -> Partitions]
  /\ IF PartitionValid(part')
     THEN /\ phase' = "TASK_AUDITED" /\ rejectReason' = "none"
     ELSE /\ phase' = "REJECTED" /\ rejectReason' = "partition-exclusivity"
  /\ UNCHANGED <<sourceOk, recOf, winPart, fitScope, pluginOk,
                 trained, evaluated, bundleOk>>

\* t3: window extraction and preprocessing fit
\*     (guards: window inheritance for INV-2, fit scope for INV-3)
Prepare ==
  /\ phase = "TASK_AUDITED"
  /\ recOf' \in [Windows -> Recordings]
  /\ winPart' \in [Windows -> Partitions]
  /\ fitScope' \in SUBSET Partitions
  /\ IF /\ WindowInheritance(part, recOf', winPart')
        /\ FitScopeValid(fitScope')
     THEN /\ phase' = "DATA_PREPARED" /\ rejectReason' = "none"
     ELSE /\ phase' = "REJECTED"
          /\ rejectReason' =
               IF FitScopeValid(fitScope')
               THEN "partition-exclusivity"
               ELSE "test-isolation"
  /\ UNCHANGED <<sourceOk, part, pluginOk, trained, evaluated, bundleOk>>

\* t4: plugin contract validation (guard: INV-6)
Construct ==
  /\ phase = "DATA_PREPARED"
  /\ \E ok \in BOOLEAN :
       /\ pluginOk' = ok
       /\ IF ok
          THEN /\ phase' = "MODEL_CONSTRUCTED" /\ rejectReason' = "none"
          ELSE /\ phase' = "REJECTED" /\ rejectReason' = "plugin-contract"
  /\ UNCHANGED <<sourceOk, part, recOf, winPart, fitScope,
                 trained, evaluated, bundleOk>>

\* t5: training one seed. The injected fault (M-LEAK-04) attempts to widen the
\*     fit scope to the test partition; the kernel rejects it fail closed and
\*     no data flows (fitScope is left untouched on the rejected path).
TrainStep(s) ==
  /\ phase = "MODEL_CONSTRUCTED"
  /\ s \in Seeds
  /\ s \notin trained
  /\ \E attemptTest \in BOOLEAN :
       /\ IF attemptTest
          THEN /\ phase' = "REJECTED" /\ rejectReason' = "test-isolation"
               /\ trained' = trained
          ELSE /\ phase' = "MODEL_CONSTRUCTED" /\ rejectReason' = "none"
               /\ trained' = trained \cup {s}
       /\ UNCHANGED <<sourceOk, part, recOf, winPart, fitScope, pluginOk,
                      evaluated, bundleOk>>

\* t6: test evaluation of one already-trained seed (test read only here)
EvaluateStep(s) ==
  /\ phase = "MODEL_CONSTRUCTED"
  /\ s \in Seeds
  /\ s \in trained
  /\ s \notin evaluated
  /\ evaluated' = evaluated \cup {s}
  /\ UNCHANGED <<phase, sourceOk, part, recOf, winPart, fitScope, pluginOk,
                 trained, bundleOk, rejectReason>>

\* t7: result-bundle validation (guards: INV-4 bijection, INV-5 provenance)
Validate ==
  /\ phase = "MODEL_CONSTRUCTED"
  /\ trained = Seeds
  /\ evaluated = Seeds
  /\ \E ok \in BOOLEAN :
       /\ bundleOk' = ok
       /\ IF ok
          THEN /\ phase' = "RESULT_VALIDATED" /\ rejectReason' = "none"
          ELSE /\ phase' = "REJECTED" /\ rejectReason' = "provenance-completeness"
  /\ UNCHANGED <<sourceOk, part, recOf, winPart, fitScope, pluginOk,
                 trained, evaluated>>

\* t8: immutable publication
Publish ==
  /\ phase = "RESULT_VALIDATED"
  /\ phase' = "PUBLISHED"
  /\ UNCHANGED <<sourceOk, part, recOf, winPart, fitScope, pluginOk,
                 trained, evaluated, bundleOk, rejectReason>>

TerminalLoop ==
  /\ phase \in {"PUBLISHED", "REJECTED"}
  /\ UNCHANGED vars

Next ==
  \/ Acquire
  \/ Audit
  \/ Prepare
  \/ Construct
  \/ \E s \in Seeds : TrainStep(s) \/ EvaluateStep(s)
  \/ Validate
  \/ Publish
  \/ TerminalLoop

Spec == Init /\ [][Next]_vars

\* ------------------------------------------------------------- invariants

\* INV-1: after acquisition, committed phases require verified sources
Inv1SourceIntegrity == phase \notin {"EMPTY", "REJECTED"} => sourceOk

\* INV-2: committed phases have valid partitions; once windows exist they
\*         must inherit their recording's partition
Inv2PartitionExclusivity == Committed => PartitionValid(part)

Inv2WindowInheritance ==
  phase \in {"DATA_PREPARED", "MODEL_CONSTRUCTED",
             "RESULT_VALIDATED", "PUBLISHED"}
    => WindowInheritance(part, recOf, winPart)

\* INV-3: committed phases after preparation never fit on test data, and the
\*         test view is only materialized for already-trained seeds
Inv3TestIsolation ==
  phase \in {"DATA_PREPARED", "MODEL_CONSTRUCTED",
             "RESULT_VALIDATED", "PUBLISHED"}
    => /\ FitScopeValid(fitScope)
       /\ evaluated \subseteq trained
       /\ trained \subseteq Seeds

\* INV-4: results are validated only when the seed set bijects with runs
Inv4SeedCompleteness ==
  phase \in {"RESULT_VALIDATED", "PUBLISHED"} => trained = Seeds /\ evaluated = Seeds

\* INV-5: validated results carry complete, hash-consistent provenance
Inv5Provenance == phase \in {"RESULT_VALIDATED", "PUBLISHED"} => bundleOk

\* INV-6: committed execution phases require a contract-valid plugin
Inv6PluginContract ==
  phase \in {"MODEL_CONSTRUCTED", "RESULT_VALIDATED", "PUBLISHED"} => pluginOk

\* Fail-closed discipline: every rejection carries a reason
FailClosed == phase = "REJECTED" => rejectReason # "none"

\* A committed experiment can always still reach publication or rejection
TypeOK ==
  /\ phase \in {"EMPTY", "SOURCES_VERIFIED", "TASK_AUDITED", "DATA_PREPARED",
                "MODEL_CONSTRUCTED", "RESULT_VALIDATED", "PUBLISHED", "REJECTED"}
  /\ fitScope \in SUBSET Partitions
  /\ trained \subseteq Seeds
  /\ evaluated \subseteq Seeds

=============================================================================
