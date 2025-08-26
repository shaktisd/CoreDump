import math
import random
import argparse
from copy import deepcopy
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------
# Sudoku generator & solver
# Ported with same algorithm and constraints as Swift
# ---------------------------

ALL_DIGITS = list(range(1, 10))


def bit(v: int) -> int:
    return 1 << (v - 1)


def box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + (col // 3)


def build_masks(grid: List[List[int]]) -> Tuple[List[int], List[int], List[int]]:
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v != 0:
                b = bit(v)
                rows[r] |= b
                cols[c] |= b
                boxes[box_index(r, c)] |= b
    return rows, cols, boxes


def first_empty_cell(grid: List[List[int]]):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None


def fill_grid_rec(grid, rows, cols, boxes):
    cell = first_empty_cell(grid)
    if cell is None:
        return True
    r, c = cell
    nums = ALL_DIGITS[:]
    random.shuffle(nums)
    bidx = box_index(r, c)
    used = rows[r] | cols[c] | boxes[bidx]
    for num in nums:
        b = bit(num)
        if used & b:
            continue
        grid[r][c] = num
        rows[r] |= b
        cols[c] |= b
        boxes[bidx] |= b
        if fill_grid_rec(grid, rows, cols, boxes):
            return True
        grid[r][c] = 0
        rows[r] &= ~b
        cols[c] &= ~b
        boxes[bidx] &= ~b
    return False


def fill_grid(grid):
    rows, cols, boxes = build_masks(grid)
    return fill_grid_rec(grid, rows, cols, boxes)


def solve_rec(grid, solutions, limit, rows, cols, boxes):
    if solutions[0] >= limit:
        return True
    cell = first_empty_cell(grid)
    if cell is None:
        solutions[0] += 1
        return solutions[0] >= limit
    r, c = cell
    bidx = box_index(r, c)
    used = rows[r] | cols[c] | boxes[bidx]
    for num in range(1, 10):
        b = bit(num)
        if used & b:
            continue
        grid[r][c] = num
        rows[r] |= b
        cols[c] |= b
        boxes[bidx] |= b
        if solve_rec(grid, solutions, limit, rows, cols, boxes):
            return True
        grid[r][c] = 0
        rows[r] &= ~b
        cols[c] &= ~b
        boxes[bidx] &= ~b
    return False


def solve(grid: List[List[int]], limit=2) -> int:
    rows, cols, boxes = build_masks(grid)
    solutions = [0]
    solve_rec(grid, solutions, limit, rows, cols, boxes)
    return solutions[0]


def clue_count(grid):
    return sum(1 for r in grid for c in r if c != 0)


def generate_sudoku(difficulty: str) -> Tuple[List[List[int]], List[List[int]]]:
    """
    difficulty choices: very-easy, easy, medium, hard, extreme
    Matches Swift's difficulty mapping roughly.
    """
    difficulty_map = {
        "very-easy": range(46, 51),
        "easy": range(40, 46),
        "medium": range(32, 40),
        "hard": range(28, 32),
        "extreme": range(17, 28),
    }
    board = [[0] * 9 for _ in range(9)]
    fill_grid(board)
    solution = deepcopy(board)
    target_clues = difficulty_map.get(difficulty, range(32, 40))
    puzzle = deepcopy(board)
    cells = list(range(81))
    random.shuffle(cells)
    cursor = 0
    clues = 81
    while cursor < len(cells) and clues > max(target_clues):
        idx = cells[cursor]
        cursor += 1
        r = idx // 9
        c = idx % 9
        backup = puzzle[r][c]
        puzzle[r][c] = 0
        test = deepcopy(puzzle)
        sol_count = solve(test, limit=2)
        if sol_count != 1:
            puzzle[r][c] = backup
        else:
            clues -= 1
    if clues > min(target_clues):
        for j in range(cursor, len(cells)):
            if clues <= min(target_clues):
                break
            idx = cells[j]
            r = idx // 9
            c = idx % 9
            backup = puzzle[r][c]
            puzzle[r][c] = 0
            test = deepcopy(puzzle)
            sol_count = solve(test, limit=2)
            if sol_count != 1:
                puzzle[r][c] = backup
            else:
                clues -= 1
    return puzzle, solution


def sudoku_board_string(board: List[List[int]]) -> str:
    horizontal = "+-------+-------+-------+"
    res = horizontal + "\n"
    for i, row in enumerate(board):
        line = "|"
        for j, v in enumerate(row):
            display = "." if v == 0 else str(v)
            line += " " + display
            if (j + 1) % 3 == 0:
                line += " |"
        res += line + "\n"
        if (i + 1) % 3 == 0:
            res += horizontal + "\n"
    return res.strip()


# ---------------------------
# Basic building blocks (RMSNorm, Rotary, SwiGLU, Attention)
# ---------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(mean_sq + self.eps)
        return x_normed * self.scale


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_len, dim]
        self.register_buffer("cos", emb.cos().unsqueeze(1))  # shape [max_len,1,dim]
        self.register_buffer("sin", emb.sin().unsqueeze(1))

    @staticmethod
    def rotate_half(x):
        # x: [..., dim], dim even
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        # x: [B, seq, num_heads, head_dim]
        seq_len = x.shape[1]
        cos = self.cos[:seq_len]  # [seq, 1, dim]
        sin = self.sin[:seq_len]

        # reshape for broadcasting: [1, seq, 1, head_dim]
        cos = cos.squeeze(1).unsqueeze(0).unsqueeze(2)
        sin = sin.squeeze(1).unsqueeze(0).unsqueeze(2)

        return x * cos + self.rotate_half(x) * sin



class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=4.0):
        super().__init__()
        inter = int(expansion * dim * 2.0 / 3.0)
        # make inter a multiple of 256 roughly similar approach as Swift; but keep it >=1
        inter = max(1, (inter + 255) // 256 * 256)
        self.gate_proj = nn.Linear(dim, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        g_u = self.gate_proj(x)
        g, u = g_u.chunk(2, dim=-1)
        return self.down_proj(F.silu(g) * u)


class Attention(nn.Module):
    def __init__(self, dim, head_dim, num_heads, key_value_heads_per_head=1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.kv_heads_per_head = key_value_heads_per_head
        self.num_kv_heads = num_heads * key_value_heads_per_head
        # qkv projection to (numHeads + 2*numKeyValueHeads) * headDim
        out_dim = (num_heads + 2 * self.num_kv_heads) * head_dim
        self.qkv = nn.Linear(dim, out_dim, bias=False)
        self.out_proj = nn.Linear(self.output_size, dim, bias=False)

    def forward(self, x, rotary: RotaryPositionEmbedding = None):
        # x: [B, seq, dim]
        B, seq, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, seq, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        query = qkv[..., : self.num_heads, :]  # [B, seq, num_heads, head_dim]
        key = qkv[..., self.num_heads : self.num_heads + self.num_kv_heads, :]
        value = qkv[..., self.num_heads + self.num_kv_heads :, :]

        # optionally apply rotary to q and k
        if rotary is not None:
            # we may need to reshape to [B, seq, num_kv_heads, head_dim] for rotary
            query = rotary(query)
            key = rotary(key)

        # reshape for matmul: query -> [B, num_kv_heads, heads_per_kv, seq, head_dim]
        # we want to compute attention across key sequence dim
        # we follow Swift's transpositions: query reshaped -> [B, seq, num_kv_heads, kv_per_head, head_dim]
        q = query.view(B, seq, self.num_kv_heads, self.kv_heads_per_head, self.head_dim).permute(
            0, 2, 3, 1, 4
        )  # [B, num_kv_heads, kv_per_head, seq, head_dim]
        k = key.permute(0, 2, 1, 3).unsqueeze(2)  # [B, num_kv_heads, 1, seq, head_dim]
        v = value.permute(0, 2, 1, 3).unsqueeze(2)  # [B, num_kv_heads, 1, seq, head_dim]

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
        attn_weights = F.softmax(attn_logits.float(), dim=-1).type_as(attn_logits)
        combined = torch.matmul(attn_weights, v)  # [B, num_kv_heads, kv_per_head, seq, head_dim]
        combined = combined.permute(0, 3, 1, 2, 4).contiguous()
        combined = combined.view(B, seq, self.output_size)
        return self.out_proj(combined)


# ---------------------------
# HRMACTBlock, Reasoner, HRMACTInner (model)
# ---------------------------


class HRMACTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, expansion, norm_eps):
        super().__init__()
        head_dim = hidden_size // num_heads
        self.self_attn = Attention(hidden_size, head_dim, num_heads)
        self.mlp = SwiGLU(hidden_size, expansion=expansion)
        self.norm_eps = norm_eps
        self.rms1 = RMSNorm(hidden_size, eps=norm_eps)
        self.rms2 = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, x, rotary):
        x = x + self.self_attn(x, rotary)
        x = self.rms1(x)
        x = x + self.mlp(x)
        x = self.rms2(x)
        return x


class HRMACTReasoner(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, expansion, norm_eps):
        super().__init__()
        self.blocks = nn.ModuleList(
            [HRMACTBlock(hidden_size, num_heads, expansion, norm_eps) for _ in range(num_layers)]
        )

    def forward(self, hidden_state, input_injection, rotary):
        x = hidden_state + input_injection
        for b in self.blocks:
            x = b(x, rotary)
        return x


class HRMACTInner(nn.Module):
    class InitialHiddenStates:
        def __init__(self, hidden_size, device, dtype):
            # initialize as trainable parameters in Swift were frozen; we'll store as buffers
            # In Swift they were frozen, but saving/loading needs them present. We'll register as buffers via module.
            self.high = torch.randn(hidden_size, dtype=dtype, device=device) * 1.0
            self.low = torch.randn(hidden_size, dtype=dtype, device=device) * 1.0

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        d = config["transformers"]["hiddenSize"]
        self.cls_token = nn.Parameter(
            torch.randn(d, dtype=torch.float32) * (1.0 / math.sqrt(d)), requires_grad=False
        )
        self.input_embedding = nn.Embedding(config["vocabSize"], d)
        self.output_head = nn.Linear(d, config["vocabSize"], bias=False)
        # qACT head: initialize bias to -5 similar to Swift to favor continue initially
        self.qACT_head = nn.Linear(d, 2)
        nn.init.constant_(self.qACT_head.bias, -5.0)
        self.rotary = RotaryPositionEmbedding(
            dim=d // config["transformers"]["numHeads"],
            max_len=config["seqLen"] + 1,
            base=config["transformers"]["ropeTheta"],
        )
        tcfg = config["transformers"]
        self.high_reasoner = HRMACTReasoner(
            num_layers=tcfg["numLayers"],
            hidden_size=tcfg["hiddenSize"],
            num_heads=tcfg["numHeads"],
            expansion=tcfg["expansion"],
            norm_eps=tcfg["normEpsilon"],
        )
        self.low_reasoner = HRMACTReasoner(
            num_layers=tcfg["numLayers"],
            hidden_size=tcfg["hiddenSize"],
            num_heads=tcfg["numHeads"],
            expansion=tcfg["expansion"],
            norm_eps=tcfg["normEpsilon"],
        )
        # initial hidden states (frozen)
        self.initial_states = HRMACTInner.InitialHiddenStates(d, device, torch.float32)
        self.register_buffer("initial_high", self.initial_states.high)
        self.register_buffer("initial_low", self.initial_states.low)

    def initial_hidden_states(self, batch_size, device):
        # return shapes similar to Swift: high and low as [batch, 1, hidden]
        return {
            "high": self.initial_high.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1).clone(),
            "low": self.initial_low.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1).clone(),
        }

    def forward(self, hidden_states: dict, inputs: torch.LongTensor):
        """
        hidden_states: dict with 'high' and 'low' keys shaped [B, 1, hidden]
        inputs: [B, seq] ints (0 for blank)
        """
        B = inputs.shape[0]
        device = inputs.device
        d = self.config["transformers"]["hiddenSize"]

        # Build embeddings: CLS + input embeddings, scaled
        cls_row = self.cls_token.to(device).unsqueeze(0).unsqueeze(0).expand(B, 1, d)
        input_emb = self.input_embedding(inputs) * math.sqrt(d)
        input_embeddings = torch.cat([cls_row, input_emb], dim=1)  # [B, seq+1, d]

        lowZ = hidden_states["low"].squeeze(1)   # [B, hidden]
        highZ = hidden_states["high"].squeeze(1) # [B, hidden]

        seq_len = input_embeddings.shape[1]

        # expand to [B, seq_len, hidden]
        lowZ = lowZ.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
        highZ = highZ.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
        
        total_cycles = self.config["highLevelCycles"] * self.config["lowLevelCycles"] - 1
        for cycle in range(1, total_cycles + 1):
            # low updates every time
            lowZ = self.low_reasoner(lowZ, highZ + input_embeddings, self.rotary)
            if cycle % self.config["lowLevelCycles"] == 0:
                highZ = self.high_reasoner(highZ, lowZ, self.rotary)

        # detach low and high (stopGradient)
        lowZ = lowZ.detach()
        highZ = highZ.detach()

        # final two passes with gradient
        lowZ = self.low_reasoner(lowZ, highZ + input_embeddings, self.rotary)
        highZ = self.high_reasoner(highZ, lowZ, self.rotary)

        # output logits computed from highZ tokens excluding CLS (positions 1..)
        output_logits = self.output_head(highZ[:, 1:, :])  # [B, seq, vocab]
        qACT_logits = self.qACT_head(highZ[:, 0, :])  # [B, 2]

        # return similar Output structure
        out = {
            # keep only CLS token as next hidden state (shape [B, 1, hidden])
            "hidden_states": {
                "high": highZ[:, 0, :].detach().unsqueeze(1),
                "low": lowZ[:, 0, :].detach().unsqueeze(1),
            },
            "output": output_logits,
            "qACT_halt": qACT_logits[:, 0],
            "qACT_continue": qACT_logits[:, 1],
        }
        return out


# ---------------------------
# TrainingBatch, loss, step
# ---------------------------


class TrainingBatch:
    DIFFICULTIES = ["easy", "medium", "hard", "extreme"]
    CURRICULUM_PROBAS = [
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.5, 0.4, 0.1, 0.0],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.3, 0.4, 0.2],
    ]

    def __init__(self, initial_hidden_states: dict, size: int, device, curriculum_level=0):
        self.device = device
        self.initial_hidden_states = initial_hidden_states
        self.size = size
        self.curriculum_level = curriculum_level
        self.total_puzzles = 0

        self.hidden_states = {
            "high": torch.zeros(size, 1, initial_hidden_states["high"].shape[-1], device=device),
            "low": torch.zeros(size, 1, initial_hidden_states["low"].shape[-1], device=device),
        }
        self.board_inputs = torch.zeros(size, 81, dtype=torch.long, device=device)
        self.board_targets = torch.zeros(size, 81, dtype=torch.long, device=device)
        self.segments = torch.zeros(size, dtype=torch.long, device=device)

        for i in range(size):
            self.replace(i)

    def sample_difficulty(self):
        probs = TrainingBatch.CURRICULUM_PROBAS[self.curriculum_level]
        r = random.random()
        cum = 0.0
        for idx, p in enumerate(probs):
            cum += p
            if r < cum:
                return TrainingBatch.DIFFICULTIES[idx]
        return TrainingBatch.DIFFICULTIES[-1]

    def replace(self, idx):
        # reset hidden states to initial
        self.hidden_states["high"][idx] = self.initial_hidden_states["high"][0, 0]
        self.hidden_states["low"][idx] = self.initial_hidden_states["low"][0, 0]
        self.segments[idx] = 0
        puzzle, solution = generate_sudoku(self.sample_difficulty())
        flat_p = [v for row in puzzle for v in row]
        flat_s = [v for row in solution for v in row]
        self.board_inputs[idx] = torch.tensor(flat_p, dtype=torch.long, device=self.device)
        self.board_targets[idx] = torch.tensor(flat_s, dtype=torch.long, device=self.device)
        self.total_puzzles += 1

    def graduate(self):
        if self.curriculum_level + 1 < len(TrainingBatch.CURRICULUM_PROBAS):
            self.curriculum_level += 1
            print(f"Graduated to level {self.curriculum_level}")
        else:
            print("Already max curriculum level")


def sudoku_loss(model: HRMACTInner, hidden_states: dict, board_inputs, board_targets, segments, key=None):
    # forward pass
    out = model(hidden_states, board_inputs)
    logits = out["output"]  # [B, seq, vocab]
    B = logits.shape[0]
    vocab = logits.shape[-1]
    # compute cross-entropy only on blanks (board_inputs == 0)
    mask = (board_inputs == 0)  # True where we need to predict
    # reshape for CE: [B*seq, vocab], targets [B*seq]
    logits_flat = logits.view(-1, vocab)
    targets_flat = board_targets.view(-1)
    mask_flat = mask.view(-1)
    if mask_flat.sum() > 0:
        loss_ce = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat] - 1)  # targets 1..9 -> 0..8
    else:
        loss_ce = torch.tensor(0.0, device=logits.device)

    # output accuracy: full board correct (or non-empty squares)
    preds = logits.argmax(dim=-1) + 1  # predicted digits 1..9
    full_correct = ((preds == board_targets) | (board_inputs != 0)).all(dim=1)  # [B]
    qACT_halt_target = full_correct.long().to(torch.float32)

    # ACT exploration logic and min halt segments logic
    next_segments = segments + 1
    is_last_segment = next_segments > model.config["act"]["haltMaxSteps"]
    q_halt = out["qACT_halt"]
    q_cont = out["qACT_continue"]
    is_halted = is_last_segment | (q_halt > q_cont)

    # exploration randomness
    if key is None:
        exploration = torch.rand(q_halt.shape, device=q_halt.device) < model.config["act"]["haltExplorationProbability"]
        min_halt_segments = torch.randint(2, model.config["act"]["haltMaxSteps"] + 1, segments.shape, device=q_halt.device) * exploration.long()
    else:
        exploration = torch.rand(q_halt.shape, device=q_halt.device) < model.config["act"]["haltExplorationProbability"]
        min_halt_segments = torch.randint(2, model.config["act"]["haltMaxSteps"] + 1, segments.shape, device=q_halt.device) * exploration.long()

    is_halted = is_halted & (next_segments > min_halt_segments)

    # compute next-step q values (stop gradient)
    next_out = model(out["hidden_states"], board_inputs)
    next_q_halt = next_out["qACT_halt"].detach()
    next_q_cont = next_out["qACT_continue"].detach()

    qACT_continue_target = torch.sigmoid(torch.where(is_last_segment, next_q_halt, torch.max(next_q_halt, next_q_cont)))

    # binary cross entropy with logits (Swift used logits version)
    bce = F.binary_cross_entropy_with_logits
    q_halt_loss = bce(q_halt, qACT_halt_target)
    q_cont_loss = bce(q_cont, qACT_continue_target)
    qact_loss = (q_halt_loss + q_cont_loss) / 2.0

    # average output loss normalized by number of predicted cells (like Swift)
    # But we used CE over selected elements -> already averaged; keep consistent:
    # We'll compute mean over masked positions
    if mask.sum() > 0:
        avg_output_loss = loss_ce
    else:
        avg_output_loss = torch.tensor(0.0, device=logits.device)

    avg_qact_loss = qact_loss
    total_loss = avg_output_loss + avg_qact_loss

    avg_output_full_accuracy = full_correct.float().mean()
    avg_qact_halt_acc = ((torch.sigmoid(q_halt) >= 0.5).long().float() == qACT_halt_target).float().mean()

    return {
        "total_loss": total_loss,
        "avg_output_loss": avg_output_loss.detach(),
        "avg_qact_loss": avg_qact_loss.detach(),
        "is_halted": is_halted.detach(),
        "avg_output_full_accuracy": avg_output_full_accuracy.detach(),
        "avg_qact_halt_accuracy": avg_qact_halt_acc.detach(),
        "next_high": out["hidden_states"]["high"].detach(),
        "next_low": out["hidden_states"]["low"].detach(),
    }


def step(model: HRMACTInner, optimizer, batch: TrainingBatch, device):
    model.train()
    # prepare inputs for autograd: use current hidden states and board inputs
    hidden_high = batch.hidden_states["high"].to(device)
    hidden_low = batch.hidden_states["low"].to(device)
    inputs = batch.board_inputs.to(device)
    targets = batch.board_targets.to(device)
    segments = batch.segments.to(device)

    # compute loss and grads (valueAndGrad equivalent)
    hidden_states_in = {"high": hidden_high, "low": hidden_low}
    # zero grads
    optimizer.zero_grad()
    loss_dict = sudoku_loss(model, hidden_states_in, inputs, targets, segments)
    loss = loss_dict["total_loss"]
    loss.backward()
    optimizer.step()

    # update batch hidden states with next detached states (deep supervision)
    batch.hidden_states["high"] = loss_dict["next_high"]
    batch.hidden_states["low"] = loss_dict["next_low"]
    batch.segments = batch.segments + 1

    # replace samples that halted
    is_halted = loss_dict["is_halted"].cpu().numpy()
    for idx, halted in enumerate(is_halted):
        if halted:
            batch.replace(idx)

    return (loss_dict["avg_output_loss"].item(), loss_dict["avg_output_full_accuracy"].item(),
            loss_dict["avg_qact_loss"].item(), loss_dict["avg_qact_halt_accuracy"].item())


# ---------------------------
# Config and main train/infer loops
# ---------------------------

DEFAULT_CONFIG = {
    "seqLen": 9 * 9,
    "vocabSize": 10,  # tokens 0..9 (0 is blank)
    "highLevelCycles": 2,
    "lowLevelCycles": 2,
    "transformers": {
        "numLayers": 2,  # keep small for quick testing; the Swift used 4
        "hiddenSize": 128,  # smaller to run on CPU; change to 256 for parity
        "numHeads": 4,
        "expansion": 4.0,
        "normEpsilon": 1e-5,
        "ropeTheta": 10000.0,
    },
    "act": {
        "haltMaxSteps": 16,
        "haltExplorationProbability": 0.1,
    },
    "dtype": torch.float32,
    "device": "cpu",
    "batchSize": 64,
}


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    config = DEFAULT_CONFIG.copy()
    config["device"] = device
    config["transformers"] = DEFAULT_CONFIG["transformers"]
    model = HRMACTInner(config, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    batch = TrainingBatch(model.initial_hidden_states(1, device) if False else model.initial_hidden_states(1, device), size=config["batchSize"], device=device)
    # Actually we need to pass the proper initial hidden states for replacement; pack them:
    initial_hidden = {"high": model.initial_hidden_states(1, device)["high"], "low": model.initial_hidden_states(1, device)["low"]}
    # reinit batch with proper initial hidden states
    batch = TrainingBatch(initial_hidden, size=config["batchSize"], device=device)

    step_idx = 0
    steps_since_grad = 0
    accuracy_history = [0.0] * 300
    try:
        while True:
            step_idx += 1
            steps_since_grad += 1
            print(f"Step {step_idx}")
            avg_out_loss, out_acc, avg_qact_loss, qact_acc = step(model, optimizer, batch, device)
            print(f"Output [{avg_out_loss:.4f} {out_acc:.4f}] | Q-ACT [{avg_qact_loss:.4f} {qact_acc:.4f}] | Puzzles [{batch.total_puzzles}] | Curriculum Level [{batch.curriculum_level}]")
            accuracy_history.pop(0)
            accuracy_history.append(out_acc)
            avg_roll = sum(accuracy_history) / len(accuracy_history)
            if avg_roll >= 0.85 and steps_since_grad >= 300:
                steps_since_grad = 0
                batch.graduate()

            # Save checkpoint periodically
            if step_idx == 1 or step_idx % 250 == 0:
                torch.save(model.state_dict(), f"checkpoint-{step_idx}.pt")
                print(f"Saved checkpoint checkpoint-{step_idx}.pt")
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint 'checkpoint-final.pt'")
        torch.save(model.state_dict(), "checkpoint-final.pt")


def infer_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    config = DEFAULT_CONFIG.copy()
    config["device"] = device
    model = HRMACTInner(config, device).to(device)
    # load checkpoint
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    puzzle, solution = generate_sudoku(args.difficulty if args.difficulty else "medium")
    print("Puzzle:\n", sudoku_board_string(puzzle))
    print("Solution:\n", sudoku_board_string(solution))
    # prepare input
    flat = [v for row in puzzle for v in row]
    puzzle_in = torch.tensor(flat, dtype=torch.long, device=device).unsqueeze(0)
    hidden = model.initial_hidden_states(1, device)
    for seg in range(1, model.config["act"]["haltMaxSteps"] + 1):
        print(f"\nSegment {seg}")
        out = model(hidden, puzzle_in)
        hidden = out["hidden_states"]
        preds = (out["output"].argmax(dim=-1).squeeze(0) + 1).cpu().numpy().tolist()
        accurate = 0
        predicted_count = 0
        predicted_flat_board = []
        for i, (p_orig, sol) in enumerate(zip([v for row in puzzle for v in row], [v for row in solution for v in row])):
            pred = preds[i]
            if p_orig != 0:
                predicted_flat_board.append(p_orig)
            else:
                accurate += 1 if pred == sol else 0
                predicted_count += 1
                predicted_flat_board.append(int(pred))
        predicted_board = [predicted_flat_board[i : i + 9] for i in range(0, 81, 9)]
        print(f"Predicted solution ({accurate}/{predicted_count}):\n{ sudoku_board_string(predicted_board) }")
        q_halt = torch.sigmoid(out["qACT_halt"][0]).item()
        q_cont = torch.sigmoid(out["qACT_continue"][0]).item()
        print(f"Q (halt - continue): {q_halt:.4f} - {q_cont:.4f}")
        if q_halt > q_cont:
            print("Halting.")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("difficulty", nargs="?", default="medium")
    parser.add_argument("--force-cpu", action="store_true", help="Do not use GPU even if available")
    args = parser.parse_args()
    if args.mode == "train":
        train_loop(args)
    elif args.mode == "infer":
        if args.checkpoint is None:
            print("Please give a checkpoint path to infer")
            return
        args.checkpoint = args.checkpoint
        args.difficulty = args.difficulty
        args.force_cpu = args.force_cpu
        infer_loop(args)


if __name__ == "__main__":
    main()
