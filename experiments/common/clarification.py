from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

InstructionStyle = Literal["oracle", "coarsened"]
AmbiguityType = Literal["goal", "process", "success"]


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


@dataclass(frozen=True)
class ClarificationTurn:
    role: Literal["user", "assistant"]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "text": self.text}


@dataclass(frozen=True)
class ClarificationResult:
    instruction: str
    clarification_transcript: List[ClarificationTurn]
    clarification_budget_turns: int
    clarification_tokens: int
    clarification_lat_ms: float = 0.0
    instruction_style: InstructionStyle = "coarsened"
    ambiguity_type: AmbiguityType = "goal"
    meta: Dict[str, Any] = field(default_factory=dict)

    def transcript_dicts(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.clarification_transcript]


def build_pointnav_instruction(
    *,
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    style: InstructionStyle,
    clarification_budget_turns: int,
    ambiguity_type: AmbiguityType = "goal",
    success_distance_m: float = 0.2,
    token_counter: Callable[[str], int] = _approx_tokens,
) -> ClarificationResult:
    """Deterministic PointNav instruction coarsening + clarification protocol.

    - `style="oracle"` embeds start/goal coordinates in the instruction (no clarification needed).
    - `style="coarsened"` hides coordinates; clarification turns reveal them in order.
    """

    clarification_budget_turns = max(0, int(clarification_budget_turns))
    transcript: List[ClarificationTurn] = []

    success_distance_m = float(success_distance_m)

    if style == "oracle":
        instruction = (
            f"Navigate from start position ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}) "
            f"to goal position ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}). "
            f"Stop when within {success_distance_m:.2f}m of the goal."
        )
        clarification_tokens = 0
        return ClarificationResult(
            instruction=instruction,
            clarification_transcript=transcript,
            clarification_budget_turns=clarification_budget_turns,
            clarification_tokens=clarification_tokens,
            clarification_lat_ms=0.0,
            instruction_style=style,
            ambiguity_type=ambiguity_type,
            meta={"success_distance_m": success_distance_m},
        )

    # Coarsened base instruction + ambiguity profile
    if ambiguity_type == "goal":
        instruction = "Navigate to the goal location."
    elif ambiguity_type == "process":
        instruction = "Navigate to the goal location. Choose an appropriate route."
    else:
        instruction = "Navigate to the goal location and stop when you are close enough."

    # Deterministic clarification protocol (0/1/2 turns).
    if ambiguity_type == "goal":
        if clarification_budget_turns >= 1:
            transcript.append(ClarificationTurn(role="user", text="Where is the goal?"))
            transcript.append(
                ClarificationTurn(
                    role="assistant",
                    text=f"Goal position is ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}).",
                )
            )
        if clarification_budget_turns >= 2:
            transcript.append(ClarificationTurn(role="user", text="Where do I start from?"))
            transcript.append(
                ClarificationTurn(
                    role="assistant",
                    text=f"Start position is ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}).",
                )
            )
    elif ambiguity_type == "process":
        if clarification_budget_turns >= 1:
            transcript.append(
                ClarificationTurn(
                    role="user",
                    text="Should I prioritize the shortest path or a safer route?",
                )
            )
            transcript.append(
                ClarificationTurn(
                    role="assistant",
                    text="Prioritize a safer route even if it is slightly longer.",
                )
            )
        if clarification_budget_turns >= 2:
            transcript.append(ClarificationTurn(role="user", text="Should I avoid sharp turns?"))
            transcript.append(
                ClarificationTurn(role="assistant", text="Yes. Avoid sharp turns when possible.")
            )
    else:  # success
        if clarification_budget_turns >= 1:
            transcript.append(ClarificationTurn(role="user", text="What counts as reaching the goal?"))
            transcript.append(
                ClarificationTurn(
                    role="assistant",
                    text=f"Consider the goal reached when you are within {success_distance_m:.2f}m.",
                )
            )
        if clarification_budget_turns >= 2:
            transcript.append(ClarificationTurn(role="user", text="Should I stop immediately once reached?"))
            transcript.append(
                ClarificationTurn(role="assistant", text="Yes. Stop immediately once the condition is met.")
            )

    clarification_tokens = sum(token_counter(t.text) for t in transcript) if transcript else 0
    return ClarificationResult(
        instruction=instruction,
        clarification_transcript=transcript,
        clarification_budget_turns=clarification_budget_turns,
        clarification_tokens=clarification_tokens,
        clarification_lat_ms=0.0,
        instruction_style=style,
        ambiguity_type=ambiguity_type,
        meta={"success_distance_m": success_distance_m},
    )


def build_text_instruction(
    *,
    oracle_instruction: str,
    style: InstructionStyle,
    clarification_budget_turns: int,
    ambiguity_type: AmbiguityType = "goal",
    coarsened_instruction: Optional[str] = None,
    token_counter: Callable[[str], int] = _approx_tokens,
) -> ClarificationResult:
    """Deterministic generic instruction coarsening + clarification protocol.

    Use this for domains where the underlying task is described as free-form text (e.g., RoboFactory).

    Design intent:
    - `style="oracle"`: return the fully specified instruction; no clarification needed.
    - `style="coarsened"`: return a high-level ambiguous instruction; clarification turns reveal details.

    NOTE: clarification "answers" may include the oracle instruction verbatim (as a deterministic ground-truth
    disclosure), so clarification token cost can be significant for long task texts.
    """

    oracle_instruction = str(oracle_instruction).strip()
    clarification_budget_turns = max(0, int(clarification_budget_turns))

    transcript: List[ClarificationTurn] = []

    if style == "oracle":
        return ClarificationResult(
            instruction=oracle_instruction,
            clarification_transcript=transcript,
            clarification_budget_turns=clarification_budget_turns,
            clarification_tokens=0,
            clarification_lat_ms=0.0,
            instruction_style=style,
            ambiguity_type=ambiguity_type,
            meta={"oracle_instruction": oracle_instruction},
        )

    # Coarsened base instruction per ambiguity profile.
    if coarsened_instruction is not None and str(coarsened_instruction).strip():
        instruction = str(coarsened_instruction).strip()
    elif ambiguity_type == "goal":
        instruction = "Complete the task objective."
    elif ambiguity_type == "process":
        instruction = "Complete the task. Choose an appropriate procedure."
    else:
        instruction = "Complete the task and stop when it is done properly."

    # Deterministic clarification protocol (0/1/2 turns).
    if clarification_budget_turns >= 1:
        if ambiguity_type == "goal":
            q = "What exactly is the goal?"
            a = f"Goal details: {oracle_instruction}"
        elif ambiguity_type == "process":
            q = "What procedure/order should I follow?"
            a = f"Procedure details: {oracle_instruction}"
        else:
            q = "What is the success condition?"
            a = f"Success condition details: {oracle_instruction}"
        transcript.append(ClarificationTurn(role="user", text=q))
        transcript.append(ClarificationTurn(role="assistant", text=a))

    if clarification_budget_turns >= 2:
        if ambiguity_type == "goal":
            q2 = "Any constraints I should follow?"
            a2 = "Follow the task constraints as specified; prioritize safety when uncertain."
        elif ambiguity_type == "process":
            q2 = "How should I handle failures or surprises?"
            a2 = "Replan as needed; recover safely and continue toward the objective."
        else:
            q2 = "Should I stop immediately once successful?"
            a2 = "Yes. Stop immediately once the success condition is satisfied."
        transcript.append(ClarificationTurn(role="user", text=q2))
        transcript.append(ClarificationTurn(role="assistant", text=a2))

    clarification_tokens = sum(token_counter(t.text) for t in transcript) if transcript else 0
    return ClarificationResult(
        instruction=instruction,
        clarification_transcript=transcript,
        clarification_budget_turns=clarification_budget_turns,
        clarification_tokens=clarification_tokens,
        clarification_lat_ms=0.0,
        instruction_style="coarsened",
        ambiguity_type=ambiguity_type,
        meta={"oracle_instruction": oracle_instruction},
    )
