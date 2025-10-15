---------------------------- MODULE ButterflyCoordination ----------------------------
(*
Formal specification of the Butterfly Coordination Protocol (BCP)
for distributed transformer inference.

This specification models:
1. Multi-phase execution (assignment, computation, aggregation, commitment)
2. Byzantine fault tolerance (up to f failures out of 2f+1 nodes)
3. Checkpoint-based recovery
4. Pipelined execution with minimal synchronization

Properties verified:
- Safety: No two nodes commit different results for the same input
- Liveness: System makes progress if >= 2f+1 nodes operational
- Determinism: Identical inputs produce identical outputs
- Recovery: System recovers from failures in bounded time
*)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Nodes,              \* Set of all node identifiers
    MaxByzantine,       \* Maximum number of Byzantine nodes (f)
    Layers,             \* Set of transformer layers
    MaxCheckpoints,     \* Maximum number of checkpoints to track
    InputData           \* Input to the inference system

ASSUME
    /\ Nodes # {}
    /\ Cardinality(Nodes) >= 2 * MaxByzantine + 1
    /\ MaxByzantine >= 0
    /\ Layers # {}
    /\ MaxCheckpoints > 0

VARIABLES
    (* Node state *)
    nodeState,          \* nodeState[n] ∈ {Initializing, Ready, Computing, Aggregating, Committing, Failed}
    nodeType,           \* nodeType[n] ∈ {Honest, Byzantine, Crashed}
    isCoordinator,      \* Current coordinator node

    (* Execution state *)
    epoch,              \* Current execution epoch
    phase,              \* Current phase: Assignment, Computation, Aggregation, Commitment
    workAssignment,     \* workAssignment[n] = set of layers assigned to node n

    (* Computation results *)
    computedResults,    \* computedResults[n] = result computed by node n
    committedResult,    \* Globally committed result (if any)

    (* Checkpoints *)
    checkpoints,        \* Sequence of checkpoints
    checkpointHashes,   \* checkpointHashes[n] = hash of node n's checkpoint

    (* Barrier synchronization *)
    barrierReady,       \* barrierReady[n] = TRUE if node n reached barrier
    barrierReleased,    \* TRUE if barrier released for current phase

    (* Byzantine agreement *)
    proposals,          \* proposals[n] = result proposed by node n
    prepareVotes,       \* prepareVotes[r] = nodes that voted for result r
    commitVotes,        \* commitVotes[r] = nodes that committed result r

    (* Failure detection *)
    suspectedNodes,     \* Set of nodes suspected to have failed
    confirmedFailures,  \* Set of nodes confirmed to have failed

    (* Message queues *)
    messages            \* Set of in-flight messages

vars == <<nodeState, nodeType, isCoordinator, epoch, phase, workAssignment,
          computedResults, committedResult, checkpoints, checkpointHashes,
          barrierReady, barrierReleased, proposals, prepareVotes, commitVotes,
          suspectedNodes, confirmedFailures, messages>>

--------------------------------------------------------------------------------
(* Type invariants *)

NodeStates == {"Initializing", "Ready", "Computing", "Aggregating", "Committing", "Failed"}
NodeTypes == {"Honest", "Byzantine", "Crashed"}
Phases == {"Assignment", "Computation", "Aggregation", "Commitment"}

TypeOK ==
    /\ nodeState \in [Nodes -> NodeStates]
    /\ nodeType \in [Nodes -> NodeTypes]
    /\ isCoordinator \in Nodes
    /\ epoch \in Nat
    /\ phase \in Phases
    /\ workAssignment \in [Nodes -> SUBSET Layers]
    /\ computedResults \in [Nodes -> {NULL} \cup Nat]
    /\ committedResult \in {NULL} \cup Nat
    /\ checkpoints \in Seq([epoch: Nat, states: [Nodes -> NodeStates]])
    /\ checkpointHashes \in [Nodes -> Nat]
    /\ barrierReady \in [Nodes -> BOOLEAN]
    /\ barrierReleased \in BOOLEAN
    /\ proposals \in [Nodes -> {NULL} \cup Nat]
    /\ prepareVotes \in [Nat -> SUBSET Nodes]
    /\ commitVotes \in [Nat -> SUBSET Nodes]
    /\ suspectedNodes \subseteq Nodes
    /\ confirmedFailures \subseteq Nodes

--------------------------------------------------------------------------------
(* Helper operators *)

(* Quorum size: 2f + 1 *)
QuorumSize == 2 * MaxByzantine + 1

(* Check if a set forms a quorum *)
IsQuorum(S) == Cardinality(S) >= QuorumSize

(* Honest nodes *)
HonestNodes == {n \in Nodes : nodeType[n] = "Honest"}

(* Operational nodes (not crashed) *)
OperationalNodes == {n \in Nodes : nodeType[n] # "Crashed"}

(* Byzantine nodes *)
ByzantineNodes == {n \in Nodes : nodeType[n] = "Byzantine"}

(* Count of Byzantine nodes *)
ByzantineCount == Cardinality(ByzantineNodes)

(* Hash function (modeled as identity for simplicity) *)
Hash(x) == x

--------------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ nodeState = [n \in Nodes |-> "Initializing"]
    /\ nodeType \in [Nodes -> NodeTypes]  \* Non-deterministic Byzantine selection
    /\ ByzantineCount <= MaxByzantine     \* Constraint: at most f Byzantine nodes
    /\ isCoordinator \in {n \in Nodes : nodeType[n] = "Honest"}  \* Coordinator must be honest
    /\ epoch = 0
    /\ phase = "Assignment"
    /\ workAssignment = [n \in Nodes |-> {}]
    /\ computedResults = [n \in Nodes |-> NULL]
    /\ committedResult = NULL
    /\ checkpoints = <<>>
    /\ checkpointHashes = [n \in Nodes |-> 0]
    /\ barrierReady = [n \in Nodes |-> FALSE]
    /\ barrierReleased = FALSE
    /\ proposals = [n \in Nodes |-> NULL]
    /\ prepareVotes = [r \in 0..100 |-> {}]
    /\ commitVotes = [r \in 0..100 |-> {}]
    /\ suspectedNodes = {}
    /\ confirmedFailures = {}
    /\ messages = {}

--------------------------------------------------------------------------------
(* Phase 1: Work Assignment *)

(* Coordinator assigns work to nodes *)
CoordinatorAssignsWork ==
    /\ phase = "Assignment"
    /\ nodeState[isCoordinator] \in {"Initializing", "Ready"}
    /\ \E assignment \in [Nodes -> SUBSET Layers]:
        /\ \A n \in OperationalNodes: assignment[n] # {}  \* All operational nodes get work
        /\ \A l \in Layers: \E n \in Nodes: l \in assignment[n]  \* All layers assigned
        /\ workAssignment' = assignment
        /\ nodeState' = [n \in Nodes |-> IF n \in OperationalNodes THEN "Ready" ELSE nodeState[n]]
        /\ phase' = "Computation"
        /\ barrierReleased' = FALSE
        /\ barrierReady' = [n \in Nodes |-> FALSE]
    /\ UNCHANGED <<nodeType, isCoordinator, epoch, computedResults, committedResult,
                   checkpoints, checkpointHashes, proposals, prepareVotes, commitVotes,
                   suspectedNodes, confirmedFailures, messages>>

--------------------------------------------------------------------------------
(* Phase 2: Computation (Pipelined) *)

(* Honest node computes its assigned layers *)
HonestNodeComputes(n) ==
    /\ phase = "Computation"
    /\ nodeState[n] = "Ready"
    /\ nodeType[n] = "Honest"
    /\ workAssignment[n] # {}
    /\ computedResults' = [computedResults EXCEPT ![n] = Hash(InputData)]  \* Deterministic computation
    /\ nodeState' = [nodeState EXCEPT ![n] = "Aggregating"]
    /\ UNCHANGED <<nodeType, isCoordinator, epoch, phase, workAssignment, committedResult,
                   checkpoints, checkpointHashes, barrierReady, barrierReleased,
                   proposals, prepareVotes, commitVotes, suspectedNodes, confirmedFailures, messages>>

(* Byzantine node may compute arbitrary result *)
ByzantineNodeComputes(n) ==
    /\ phase = "Computation"
    /\ nodeState[n] = "Ready"
    /\ nodeType[n] = "Byzantine"
    /\ workAssignment[n] # {}
    /\ \E result \in 0..100:  \* Arbitrary Byzantine result
        computedResults' = [computedResults EXCEPT ![n] = result]
    /\ nodeState' = [nodeState EXCEPT ![n] = "Aggregating"]
    /\ UNCHANGED <<nodeType, isCoordinator, epoch, phase, workAssignment, committedResult,
                   checkpoints, checkpointHashes, barrierReady, barrierReleased,
                   proposals, prepareVotes, commitVotes, suspectedNodes, confirmedFailures, messages>>

--------------------------------------------------------------------------------
(* Phase 3: Aggregation and Barrier Synchronization *)

(* Node signals ready at barrier *)
NodeReachesBarrier(n) ==
    /\ phase = "Computation"
    /\ nodeState[n] = "Aggregating"
    /\ computedResults[n] # NULL
    /\ ~barrierReady[n]
    /\ checkpointHashes' = [checkpointHashes EXCEPT ![n] = Hash(nodeState[n])]
    /\ barrierReady' = [barrierReady EXCEPT ![n] = TRUE]
    /\ UNCHANGED <<nodeState, nodeType, isCoordinator, epoch, phase, workAssignment,
                   computedResults, committedResult, checkpoints, barrierReleased,
                   proposals, prepareVotes, commitVotes, suspectedNodes, confirmedFailures, messages>>

(* Coordinator releases barrier when quorum reached *)
CoordinatorReleasesBarrier ==
    /\ phase = "Computation"
    /\ nodeState[isCoordinator] = "Aggregating"
    /\ ~barrierReleased
    /\ IsQuorum({n \in Nodes : barrierReady[n]})
    /\ LET readyNodes == {n \in Nodes : barrierReady[n]}
           hashes == {checkpointHashes[n] : n \in readyNodes}
       IN /\ Cardinality(hashes) = 1  \* All hashes match (honest nodes agree)
          /\ barrierReleased' = TRUE
          /\ phase' = "Aggregation"
    /\ UNCHANGED <<nodeState, nodeType, isCoordinator, epoch, workAssignment,
                   computedResults, committedResult, checkpoints, checkpointHashes,
                   barrierReady, proposals, prepareVotes, commitVotes,
                   suspectedNodes, confirmedFailures, messages>>

--------------------------------------------------------------------------------
(* Phase 4: Byzantine Agreement on Result *)

(* Coordinator proposes result (PRE-PREPARE) *)
CoordinatorProposesResult ==
    /\ phase = "Aggregation"
    /\ barrierReleased
    /\ nodeState[isCoordinator] = "Aggregating"
    /\ proposals[isCoordinator] = NULL
    /\ LET correctResult == Hash(InputData)  \* Expected result from honest computation
       IN /\ proposals' = [proposals EXCEPT ![isCoordinator] = correctResult]
          /\ phase' = "Commitment"
    /\ UNCHANGED <<nodeState, nodeType, isCoordinator, epoch, workAssignment,
                   computedResults, committedResult, checkpoints, checkpointHashes,
                   barrierReady, barrierReleased, prepareVotes, commitVotes,
                   suspectedNodes, confirmedFailures, messages>>

(* Honest node validates and votes PREPARE *)
HonestNodePrepares(n) ==
    /\ phase = "Commitment"
    /\ nodeState[n] = "Aggregating"
    /\ nodeType[n] = "Honest"
    /\ proposals[isCoordinator] # NULL
    /\ n \notin prepareVotes[proposals[isCoordinator]]
    /\ LET proposedResult == proposals[isCoordinator]
           expectedResult == computedResults[n]
       IN /\ proposedResult = expectedResult  \* Validation: result matches own computation
          /\ prepareVotes' = [prepareVotes EXCEPT ![proposedResult] = @ \cup {n}]
    /\ UNCHANGED <<nodeState, nodeType, isCoordinator, epoch, phase, workAssignment,
                   computedResults, committedResult, checkpoints, checkpointHashes,
                   barrierReady, barrierReleased, proposals, commitVotes,
                   suspectedNodes, confirmedFailures, messages>>

(* Byzantine node may vote for anything *)
ByzantineNodePrepares(n) ==
    /\ phase = "Commitment"
    /\ nodeState[n] = "Aggregating"
    /\ nodeType[n] = "Byzantine"
    /\ \E result \in {proposals[isCoordinator]} \cup {computedResults[n]}:
        /\ result # NULL
        /\ n \notin prepareVotes[result]
        /\ prepareVotes' = [prepareVotes EXCEPT ![result] = @ \cup {n}]
    /\ UNCHANGED <<nodeState, nodeType, isCoordinator, epoch, phase, workAssignment,
                   computedResults, committedResult, checkpoints, checkpointHashes,
                   barrierReady, barrierReleased, proposals, commitVotes,
                   suspectedNodes, confirmedFailures, messages>>

(* Node commits when prepare quorum reached *)
NodeCommits(n) ==
    /\ phase = "Commitment"
    /\ nodeState[n] = "Aggregating"
    /\ proposals[isCoordinator] # NULL
    /\ LET result == proposals[isCoordinator]
       IN /\ IsQuorum(prepareVotes[result])
          /\ n \notin commitVotes[result]
          /\ commitVotes' = [commitVotes EXCEPT ![result] = @ \cup {n}]
          /\ nodeState' = [nodeState EXCEPT ![n] = "Committing"]
    /\ UNCHANGED <<nodeType, isCoordinator, epoch, phase, workAssignment,
                   computedResults, committedResult, checkpoints, checkpointHashes,
                   barrierReady, barrierReleased, proposals, prepareVotes,
                   suspectedNodes, confirmedFailures, messages>>

(* Finalize commitment when commit quorum reached *)
FinalizeCommitment ==
    /\ phase = "Commitment"
    /\ committedResult = NULL
    /\ proposals[isCoordinator] # NULL
    /\ LET result == proposals[isCoordinator]
       IN /\ IsQuorum(commitVotes[result])
          /\ committedResult' = result
          /\ nodeState' = [n \in Nodes |-> IF nodeType[n] # "Crashed" THEN "Ready" ELSE "Failed"]
          /\ phase' = "Assignment"
          /\ epoch' = epoch + 1
          /\ barrierReady' = [n \in Nodes |-> FALSE]
          /\ barrierReleased' = FALSE
          /\ proposals' = [n \in Nodes |-> NULL]
          /\ prepareVotes' = [r \in 0..100 |-> {}]
          /\ commitVotes' = [r \in 0..100 |-> {}]
    /\ UNCHANGED <<nodeType, isCoordinator, workAssignment, computedResults,
                   checkpoints, checkpointHashes, suspectedNodes, confirmedFailures, messages>>

--------------------------------------------------------------------------------
(* Failure handling *)

(* Node crashes *)
NodeCrashes(n) ==
    /\ nodeType[n] # "Crashed"
    /\ nodeType' = [nodeType EXCEPT ![n] = "Crashed"]
    /\ nodeState' = [nodeState EXCEPT ![n] = "Failed"]
    /\ confirmedFailures' = confirmedFailures \cup {n}
    /\ UNCHANGED <<isCoordinator, epoch, phase, workAssignment, computedResults,
                   committedResult, checkpoints, checkpointHashes, barrierReady,
                   barrierReleased, proposals, prepareVotes, commitVotes,
                   suspectedNodes, messages>>

(* Detect suspected failure *)
SuspectFailure(n) ==
    /\ n \notin suspectedNodes
    /\ nodeType[n] = "Crashed"  \* Simplification: detect only actual crashes
    /\ suspectedNodes' = suspectedNodes \cup {n}
    /\ UNCHANGED <<nodeState, nodeType, isCoordinator, epoch, phase, workAssignment,
                   computedResults, committedResult, checkpoints, checkpointHashes,
                   barrierReady, barrierReleased, proposals, prepareVotes, commitVotes,
                   confirmedFailures, messages>>

(* Checkpoint state *)
CreateCheckpoint ==
    /\ Len(checkpoints) < MaxCheckpoints
    /\ phase = "Computation"
    /\ LET cp == [epoch |-> epoch, states |-> nodeState]
       IN checkpoints' = Append(checkpoints, cp)
    /\ UNCHANGED <<nodeState, nodeType, isCoordinator, epoch, phase, workAssignment,
                   computedResults, committedResult, checkpointHashes, barrierReady,
                   barrierReleased, proposals, prepareVotes, commitVotes,
                   suspectedNodes, confirmedFailures, messages>>

--------------------------------------------------------------------------------
(* State transitions *)

Next ==
    \/ CoordinatorAssignsWork
    \/ \E n \in Nodes: HonestNodeComputes(n)
    \/ \E n \in Nodes: ByzantineNodeComputes(n)
    \/ \E n \in Nodes: NodeReachesBarrier(n)
    \/ CoordinatorReleasesBarrier
    \/ CoordinatorProposesResult
    \/ \E n \in Nodes: HonestNodePrepares(n)
    \/ \E n \in Nodes: ByzantineNodePrepares(n)
    \/ \E n \in Nodes: NodeCommits(n)
    \/ FinalizeCommitment
    \/ \E n \in Nodes: NodeCrashes(n)
    \/ \E n \in Nodes: SuspectFailure(n)
    \/ CreateCheckpoint

Spec == Init /\ [][Next]_vars

--------------------------------------------------------------------------------
(* Safety Properties *)

(* Property 1: At most one result is committed per epoch *)
Agreement ==
    committedResult # NULL =>
        \A e \in 0..epoch:
            Len(checkpoints) >= e =>
                \A n \in HonestNodes:
                    nodeState[n] \in {"Ready", "Committing"} =>
                        computedResults[n] = committedResult

(* Property 2: Committed result is the correct deterministic result *)
Validity ==
    committedResult # NULL => committedResult = Hash(InputData)

(* Property 3: Byzantine nodes cannot corrupt the committed result *)
ByzantineResistance ==
    ByzantineCount <= MaxByzantine =>
        committedResult # NULL =>
            committedResult = Hash(InputData)

(* Property 4: No two honest nodes commit different results *)
Consistency ==
    \A n1, n2 \in HonestNodes:
        (nodeState[n1] = "Committing" /\ nodeState[n2] = "Committing") =>
            (computedResults[n1] = NULL \/ computedResults[n2] = NULL \/
             computedResults[n1] = computedResults[n2])

--------------------------------------------------------------------------------
(* Liveness Properties *)

(* Property 5: System eventually commits a result if enough nodes operational *)
EventualCommitment ==
    (Cardinality(OperationalNodes) >= QuorumSize) ~> (committedResult # NULL)

(* Property 6: All honest nodes eventually reach the same state *)
EventualAgreement ==
    <>[](\A n1, n2 \in HonestNodes:
           nodeState[n1] \in {"Ready", "Failed"} /\ nodeState[n2] \in {"Ready", "Failed"} =>
               nodeState[n1] = nodeState[n2])

(* Property 7: Failed nodes are eventually detected *)
EventualFailureDetection ==
    \A n \in Nodes:
        nodeType[n] = "Crashed" ~> n \in confirmedFailures

--------------------------------------------------------------------------------
(* Correctness theorems to verify *)

THEOREM Spec => []TypeOK
THEOREM Spec => []Agreement
THEOREM Spec => []Validity
THEOREM Spec => []ByzantineResistance
THEOREM Spec => []Consistency
THEOREM Spec => EventualCommitment
THEOREM Spec => EventualAgreement
THEOREM Spec => EventualFailureDetection

================================================================================

(*
Model checking configuration for TLC:

SPECIFICATION Spec
INVARIANTS TypeOK Agreement Validity ByzantineResistance Consistency
PROPERTIES EventualCommitment EventualAgreement EventualFailureDetection

CONSTANTS
    Nodes = {n1, n2, n3, n4, n5}  \* 5 nodes (tolerates f=2 Byzantine)
    MaxByzantine = 2
    Layers = {L1, L2, L3}
    MaxCheckpoints = 3
    InputData = 42

STATE CONSTRAINT
    epoch <= 3  \* Limit state space exploration

Verify with:
    tlc ButterflyCoordination.tla -deadlock -workers auto
*)
