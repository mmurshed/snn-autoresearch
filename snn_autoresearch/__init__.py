"""SNN-AutoResearch: LLM-driven discovery of surrogate gradient functions for SNNs."""

from .candidate import SurrogateCandidate, BASELINES
from .verify import verify
from .spike import SurrogateSpike, LIFNeuron, make_spike_fn
from .llm import get_llm, parse_candidates
from .evaluate import train_and_evaluate, TrainingResult
