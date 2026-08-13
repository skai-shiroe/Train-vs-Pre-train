"""Autoregressive generation for the from scratch Transformer.

Training uses teacher forcing: the whole target is known and fed at once.
Inference cannot. The decoder starts from ``decoder_start_token_id`` and
extends its own output one token at a time, until it emits the end of sequence
token or reaches the length budget.

Three strategies are provided.

**Greedy search** takes the most likely token at each step. It is cheap and
deterministic, and it is the reference baseline: any gain reported with beam
search must be measured against it.

**Beam search** keeps the ``k`` most likely partial sequences instead of one.
Greedy search is myopic: a token that looks best now can lead to a poor
sequence. Beam search delays the commitment. Its score is the sum of the log
probabilities, divided by ``length ** length_penalty`` so that short sequences
are not mechanically favoured.

**Sampling** draws each token from the distribution rather than taking its
maximum, so two runs on the same document return two different summaries. The
draw is narrowed first, by ``temperature``, then by ``top_k``, then by the
``top_p`` nucleus; without those cuts the tail of a vocabulary of thousands of
tokens carries enough total mass to be picked regularly, and a summary that
picks from the tail is a summary that invents. It is never used to measure:
every score this project reports is produced by one of the two strategies
above. What it is for is an interface that must not answer the same sentence
twice, and :func:`src.models.generation.apply_seed` is what makes one of its
answers reproducible afterwards.

**No key value cache.** Each step re-decodes the whole prefix. With a target
budget of 64 tokens the cost stays modest and the code stays readable, which
matters more here than throughput. Adding a cache would change the speed, never
the output.

The hyperparameters live in :class:`src.models.generation.GenerationConfig`,
which the pretrained baseline reads too. Both models therefore decode under the
same constraints, and a score gap can only come from the models themselves.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.generation import GenerationConfig, apply_seed
from src.models.scratch.transformer import ScratchTransformer

NEGATIVE_INFINITY = float("-inf")


def banned_ngram_tokens(sequence: torch.Tensor, ngram_size: int) -> list[int]:
    """Return the tokens that would close a repeated n-gram.

    Args:
        sequence: One dimensional tensor holding the tokens generated so far.
        ngram_size: Size of the n-gram to forbid. Zero disables the rule.

    Returns:
        The identifiers that must not be generated next.
    """
    if ngram_size <= 0 or sequence.numel() < ngram_size:
        return []

    tokens = sequence.tolist()
    prefix = tuple(tokens[-(ngram_size - 1) :]) if ngram_size > 1 else ()

    banned: set[int] = set()
    for start in range(len(tokens) - ngram_size + 1):
        candidate = tuple(tokens[start : start + ngram_size])
        if candidate[:-1] == prefix:
            banned.add(candidate[-1])
    return sorted(banned)


def _apply_constraints(
    log_probabilities: torch.Tensor,
    generated: torch.Tensor,
    config: GenerationConfig,
    eos_token_id: int,
    step: int,
) -> torch.Tensor:
    """Forbid the tokens the decoding constraints rule out.

    Args:
        log_probabilities: Tensor of shape ``(rows, vocab_size)``.
        generated: Tokens generated so far, of shape ``(rows, length)``.
        config: Decoding configuration.
        eos_token_id: End of sequence identifier.
        step: Number of tokens generated so far.

    Returns:
        The constrained log probabilities.
    """
    constrained = log_probabilities.clone()

    if step < config.min_new_tokens:
        constrained[:, eos_token_id] = NEGATIVE_INFINITY

    if config.no_repeat_ngram_size > 0:
        for row in range(constrained.size(0)):
            banned = banned_ngram_tokens(generated[row], config.no_repeat_ngram_size)
            if banned:
                constrained[row, banned] = NEGATIVE_INFINITY

    return constrained


@torch.no_grad()
def greedy_search(
    model: ScratchTransformer,
    source_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Generate one sequence per input by taking the most likely token.

    Args:
        model: The trained model, used in evaluation mode.
        source_ids: Integer tensor of shape ``(batch, src_len)``.
        config: Decoding configuration.

    Returns:
        A tensor of shape ``(batch, generated_len)`` holding the generated
        tokens, start token excluded, padded after the end of sequence token.
    """
    model.eval()
    batch_size = source_ids.size(0)
    device = source_ids.device
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id

    memory, memory_mask, _ = model.encode(source_ids)

    decoder_input = torch.full(
        (batch_size, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(config.max_new_tokens):
        logits, _ = model.decode(decoder_input, memory, memory_mask)
        log_probabilities = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        log_probabilities = _apply_constraints(
            log_probabilities, decoder_input[:, 1:], config, eos_token_id, step
        )

        next_token = log_probabilities.argmax(dim=-1)
        # A finished sequence keeps emitting padding, so its tokens stay stable.
        next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)

        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_token_id)

        if bool(finished.all()):
            break

    return decoder_input[:, 1:]


def narrow_for_sampling(log_probabilities: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    """Apply the temperature and the two truncations, in that order.

    The order is not free. ``top_k`` and ``top_p`` select on the tempered
    distribution, so raising the temperature widens what the cuts then keep;
    truncating first would fix the candidate set before the temperature had any
    say and make the two knobs independent, which is not what a caller reading
    their names expects.

    The entries the decoding constraints already forbade hold minus infinity,
    and dividing that by a positive temperature leaves it there. A banned token
    therefore cannot come back through the sampling path.

    Args:
        log_probabilities: Constrained log probabilities of shape
            ``(rows, vocab_size)``.
        config: Decoding configuration.

    Returns:
        Scores of the same shape, with the discarded tokens at minus infinity.
        They are not renormalised: the caller runs a softmax over them.
    """
    scores = log_probabilities / config.temperature

    if config.top_k > 0:
        # A vocabulary smaller than the requested k would make topk raise.
        keep = min(config.top_k, scores.size(-1))
        floor = scores.topk(keep, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < floor, NEGATIVE_INFINITY)

    if config.top_p < 1.0:
        ordered, ordering = scores.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)

        # Shifted by one so the token that crosses the threshold is kept. Without
        # the shift a distribution whose first token already exceeds top_p would
        # have an empty nucleus, and multinomial would draw from nothing.
        discard = cumulative > config.top_p
        discard[:, 1:] = discard[:, :-1].clone()
        discard[:, 0] = False

        scores = scores.masked_fill(discard.scatter(1, ordering, discard), NEGATIVE_INFINITY)

    return scores


@torch.no_grad()
def sample_search(
    model: ScratchTransformer,
    source_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Generate one sequence per input by drawing each token from the distribution.

    The loop is the one of :func:`greedy_search`, with the ``argmax`` replaced
    by a draw over the narrowed distribution. Everything else, the length
    budget, the minimum length, the n-gram blocking and the padding of a
    finished row, behaves identically: a sampled summary must obey the same
    constraints as a greedy one, or the two would not be comparable at all.

    Args:
        model: The trained model, used in evaluation mode.
        source_ids: Integer tensor of shape ``(batch, src_len)``.
        config: Decoding configuration, with ``do_sample`` set.

    Returns:
        A tensor of shape ``(batch, generated_len)`` holding the generated
        tokens, start token excluded, padded after the end of sequence token.
    """
    model.eval()
    batch_size = source_ids.size(0)
    device = source_ids.device
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id

    memory, memory_mask, _ = model.encode(source_ids)

    decoder_input = torch.full(
        (batch_size, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(config.max_new_tokens):
        logits, _ = model.decode(decoder_input, memory, memory_mask)
        log_probabilities = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        log_probabilities = _apply_constraints(
            log_probabilities, decoder_input[:, 1:], config, eos_token_id, step
        )

        narrowed = narrow_for_sampling(log_probabilities, config)
        next_token = torch.multinomial(narrowed.softmax(dim=-1), num_samples=1).squeeze(1)
        # A finished sequence keeps emitting padding, so its tokens stay stable.
        next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)

        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_token_id)

        if bool(finished.all()):
            break

    return decoder_input[:, 1:]


@torch.no_grad()
def beam_search(
    model: ScratchTransformer,
    source_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Generate one sequence per input by keeping the best partial hypotheses.

    Args:
        model: The trained model, used in evaluation mode.
        source_ids: Integer tensor of shape ``(batch, src_len)``.
        config: Decoding configuration, with ``num_beams`` above one.

    Returns:
        A tensor of shape ``(batch, generated_len)`` holding the best sequence
        of each input, padded to the longest one.
    """
    model.eval()
    batch_size = source_ids.size(0)
    beams = config.num_beams
    device = source_ids.device
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id

    memory, memory_mask, _ = model.encode(source_ids)
    memory = memory.repeat_interleave(beams, dim=0)
    memory_mask = memory_mask.repeat_interleave(beams, dim=0)

    sequences = torch.full(
        (batch_size * beams, 1),
        model.config.decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )

    # Only the first beam of each input starts alive, otherwise the k beams
    # would all expand the same prefix and produce k identical hypotheses.
    scores = torch.full((batch_size, beams), NEGATIVE_INFINITY, device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)

    finished: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch_size)]

    for step in range(config.max_new_tokens):
        logits, _ = model.decode(sequences, memory, memory_mask)
        log_probabilities = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        log_probabilities = _apply_constraints(
            log_probabilities, sequences[:, 1:], config, eos_token_id, step
        )

        vocab_size = log_probabilities.size(-1)
        candidates = (log_probabilities + scores.unsqueeze(1)).view(batch_size, beams * vocab_size)

        # Two candidates per beam: enough to always refill the beam even when
        # every top candidate ends a sequence.
        top_scores, top_indices = candidates.topk(2 * beams, dim=-1)
        beam_indices = torch.div(top_indices, vocab_size, rounding_mode="floor")
        token_indices = top_indices % vocab_size

        next_sequences = []
        next_scores = []
        for batch in range(batch_size):
            kept = 0
            for candidate in range(2 * beams):
                if kept == beams:
                    break

                beam = int(beam_indices[batch, candidate])
                token = int(token_indices[batch, candidate])
                score = float(top_scores[batch, candidate])
                row = batch * beams + beam
                extended = torch.cat(
                    [sequences[row], torch.tensor([token], device=device, dtype=torch.long)]
                )

                if token == eos_token_id:
                    length = extended.size(0) - 1
                    finished[batch].append((score / (length**config.length_penalty), extended[1:]))
                    continue

                next_sequences.append(extended)
                next_scores.append(score)
                kept += 1

            while kept < beams:
                next_sequences.append(sequences[batch * beams])
                next_scores.append(NEGATIVE_INFINITY)
                kept += 1

        sequences = torch.stack(next_sequences)
        scores = torch.tensor(next_scores, device=device)

        if all(len(hypotheses) >= beams for hypotheses in finished):
            break

    return _select_best(finished, sequences, scores, batch_size, beams, pad_token_id)


def _select_best(
    finished: list[list[tuple[float, torch.Tensor]]],
    sequences: torch.Tensor,
    scores: torch.Tensor,
    batch_size: int,
    beams: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Pick the best hypothesis of each input and pad them to a common length.

    The alive beams are compared on their raw score, without the length
    penalty: at that point they all hold the same number of tokens, so the
    penalty is a constant factor and cannot change the ranking.

    Args:
        finished: Hypotheses that emitted the end of sequence token.
        sequences: Beams still alive when the loop stopped.
        scores: Scores of the alive beams.
        batch_size: Number of inputs.
        beams: Beam width.
        pad_token_id: Identifier used to pad the shorter sequences.

    Returns:
        A tensor of shape ``(batch, generated_len)``.
    """
    best: list[torch.Tensor] = []
    for batch in range(batch_size):
        # An input whose beams never emitted the end of sequence token falls
        # back on its best alive beam, truncated by the length budget.
        if not finished[batch]:
            offset = batch * beams
            alive = scores[offset : offset + beams].argmax()
            best.append(sequences[offset + int(alive)][1:])
            continue

        best.append(max(finished[batch], key=lambda item: item[0])[1])

    longest = max(int(candidate.size(0)) for candidate in best)
    padded = torch.full(
        (batch_size, longest), pad_token_id, dtype=torch.long, device=sequences.device
    )
    for index, candidate in enumerate(best):
        padded[index, : candidate.size(0)] = candidate
    return padded


def generate(
    model: ScratchTransformer,
    source_ids: torch.Tensor,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Generate summaries for a batch of documents.

    Args:
        model: The trained model.
        source_ids: Integer tensor of shape ``(batch, src_len)``.
        config: Decoding configuration. Defaults to greedy search over 64 tokens.

    Returns:
        A tensor of shape ``(batch, generated_len)``.
    """
    settings = config or GenerationConfig()
    apply_seed(settings)
    if settings.do_sample:
        return sample_search(model, source_ids, settings)
    if settings.num_beams == 1:
        return greedy_search(model, source_ids, settings)
    return beam_search(model, source_ids, settings)
