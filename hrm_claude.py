import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import argparse
import os


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class TransformerConfig:
    num_layers: int = 4
    hidden_size: int = 256
    num_heads: int = 4
    expansion: float = 4.0
    norm_epsilon: float = 1e-5
    rope_theta: float = 10000.0


@dataclass
class ACTConfig:
    halt_max_steps: int = 16
    halt_exploration_probability: float = 0.1


@dataclass
class HRMACTModelConfig:
    seq_len: int = 81  # 9x9 Sudoku
    vocab_size: int = 10  # 0-9 digits
    high_level_cycles: int = 2
    low_level_cycles: int = 2
    transformers: TransformerConfig = None
    act: ACTConfig = None
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.transformers is None:
            self.transformers = TransformerConfig()
        if self.act is None:
            self.act = ACTConfig()


# ============================================================================
# Utility Functions
# ============================================================================

def truncated_normal_init(shape, std=1.0, lower=-2.0, upper=2.0, device='cpu', dtype=torch.float32):
    """Truncated normal initialization"""
    if std == 0.0:
        return torch.zeros(shape, device=device, dtype=dtype)
    
    # Simple truncated normal approximation
    tensor = torch.randn(shape, device=device, dtype=dtype) * std
    return torch.clamp(tensor, lower * std, upper * std)


def rms_norm(x, epsilon=1e-6):
    """RMS Normalization"""
    original_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + epsilon)).to(original_dtype)


# ============================================================================
# Rotary Position Embedding
# ============================================================================

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_length, base=10000.0, dtype=torch.float32):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_length).float()
        freqs = torch.outer(t, inv_freq)
        
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(-2)
        self.register_buffer('cos', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin', emb.sin().to(dtype), persistent=False)

    @staticmethod
    def rotate_half(x):
        x = x.transpose(-2, -1)
        half_dim = x.shape[-1] // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1).transpose(-2, -1)

    def forward(self, x):
        return (x * self.cos) + (self.rotate_half(x) * self.sin)


# ============================================================================
# SwiGLU MLP
# ============================================================================

class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=4.0, dtype=torch.float32):
        super().__init__()
        inter_dim = self._find_multiple(int(expansion * dim * 2.0 / 3.0), 256)
        self.gate_up_proj = nn.Linear(dim, inter_dim * 2, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False, dtype=dtype)

    @staticmethod
    def _find_multiple(a, b):
        return ((a + b - 1) // b) * b

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ============================================================================
# Attention
# ============================================================================

class Attention(nn.Module):
    def __init__(self, dim, head_dim, num_heads, key_value_heads_per_head=1, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.key_value_heads_per_head = key_value_heads_per_head
        self.num_key_value_heads = num_heads * key_value_heads_per_head

        self.qkv_proj = nn.Linear(
            dim, 
            (num_heads + 2 * self.num_key_value_heads) * head_dim, 
            bias=False, 
            dtype=dtype
        )
        self.out_proj = nn.Linear(self.output_size, dim, bias=False, dtype=dtype)

    def forward(self, x, rotary_position_embedding=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x).view(
            batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if rotary_position_embedding is not None:
            query = rotary_position_embedding(query)
            key = rotary_position_embedding(key)

        # Reshape for attention
        query = query.view(
            batch_size, seq_len, self.num_key_value_heads, self.key_value_heads_per_head, self.head_dim
        ).transpose(1, 2).transpose(2, 3)  # [B, KV_H, KV_H_PER_H, S, H_D]
        
        key = key.transpose(1, 2).unsqueeze(2)  # [B, KV_H, 1, S, H_D]
        value = value.transpose(1, 2).unsqueeze(2)  # [B, KV_H, 1, S, H_D]

        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits.float(), dim=-1).to(attn_logits.dtype)
        combined = torch.matmul(attn_weights, value)
        
        combined = combined.transpose(1, 3).transpose(2, 3).contiguous().view(
            batch_size, seq_len, self.dim
        )

        return self.out_proj(combined)


# ============================================================================
# Embedding Layer
# ============================================================================

class Embedding(nn.Module):
    def __init__(self, vocab_size, dim, init_std=1.0, dtype=torch.float32):
        super().__init__()
        self.embeddings = nn.Parameter(
            truncated_normal_init([vocab_size, dim], std=init_std, dtype=dtype)
        )

    def forward(self, x):
        return self.embeddings[x]


# ============================================================================
# HRM Components
# ============================================================================

class HRMACTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, expansion, norm_epsilon, dtype=torch.float32):
        super().__init__()
        self.self_attn = Attention(
            dim=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            key_value_heads_per_head=1,
            dtype=dtype
        )
        self.mlp = SwiGLU(dim=hidden_size, expansion=expansion, dtype=dtype)
        self.norm_epsilon = norm_epsilon

    def forward(self, x, rotary_position_embedding=None):
        x = rms_norm(x + self.self_attn(x, rotary_position_embedding), self.norm_epsilon)
        x = rms_norm(x + self.mlp(x), self.norm_epsilon)
        return x


class HRMACTReasoner(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, expansion, norm_epsilon, dtype=torch.float32):
        super().__init__()
        self.blocks = nn.ModuleList([
            HRMACTBlock(hidden_size, num_heads, expansion, norm_epsilon, dtype)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_state, input_injection, rotary_position_embedding=None):
        hidden_state = hidden_state + input_injection
        for block in self.blocks:
            hidden_state = block(hidden_state, rotary_position_embedding)
        return hidden_state


# ============================================================================
# Main HRM Model
# ============================================================================

class HRMACTInner(nn.Module):
    def __init__(self, config: HRMACTModelConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.cls_token = nn.Parameter(
            truncated_normal_init(
                [config.transformers.hidden_size],
                std=1.0 / math.sqrt(config.transformers.hidden_size),
                dtype=config.dtype
            )
        )

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            dim=config.transformers.hidden_size,
            init_std=1.0 / math.sqrt(config.transformers.hidden_size),
            dtype=config.dtype
        )

        self.output_head = nn.Linear(
            config.transformers.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.dtype
        )

        self.q_act_head = nn.Linear(
            config.transformers.hidden_size,
            2,
            bias=True,
            dtype=config.dtype
        )
        # Initialize Q-ACT head bias to -5
        nn.init.constant_(self.q_act_head.bias, -5.0)
        nn.init.zeros_(self.q_act_head.weight)

        self.rotary_emb = RotaryPositionEmbedding(
            dim=config.transformers.hidden_size // config.transformers.num_heads,
            max_length=config.seq_len + 1,  # +1 for CLS token
            base=config.transformers.rope_theta,
            dtype=config.dtype
        )

        self.high_level_reasoner = HRMACTReasoner(
            num_layers=config.transformers.num_layers,
            hidden_size=config.transformers.hidden_size,
            num_heads=config.transformers.num_heads,
            expansion=config.transformers.expansion,
            norm_epsilon=config.transformers.norm_epsilon,
            dtype=config.dtype
        )

        self.low_level_reasoner = HRMACTReasoner(
            num_layers=config.transformers.num_layers,
            hidden_size=config.transformers.hidden_size,
            num_heads=config.transformers.num_heads,
            expansion=config.transformers.expansion,
            norm_epsilon=config.transformers.norm_epsilon,
            dtype=config.dtype
        )

        # Initial hidden states (learnable parameters)
        self.initial_high_level = nn.Parameter(
            truncated_normal_init([config.transformers.hidden_size], std=1.0, dtype=config.dtype)
        )
        self.initial_low_level = nn.Parameter(
            truncated_normal_init([config.transformers.hidden_size], std=1.0, dtype=config.dtype)
        )

    def get_initial_hidden_states(self, batch_size):
        return {
            'high_level': self.initial_high_level.unsqueeze(0).repeat(batch_size, 1, 1),
            'low_level': self.initial_low_level.unsqueeze(0).repeat(batch_size, 1, 1)
        }

    def forward(self, hidden_states, inputs):
        batch_size = inputs.shape[0]
        
        # Create input embeddings with CLS token
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        input_embeds = self.input_embedding(inputs)
        input_embeddings = torch.cat([cls_tokens, input_embeds], dim=1)
        input_embeddings *= math.sqrt(self.config.transformers.hidden_size)

        low_level_z = hidden_states['low_level']
        high_level_z = hidden_states['high_level']

        # Iterative reasoning cycles (stop gradients for all but last)
        total_cycles = self.config.high_level_cycles * self.config.low_level_cycles
        for cycle in range(1, total_cycles):
            low_level_z = self.low_level_reasoner(
                low_level_z,
                high_level_z + input_embeddings,
                self.rotary_emb
            )
            
            if cycle % self.config.low_level_cycles == 0:
                high_level_z = self.high_level_reasoner(
                    high_level_z,
                    low_level_z,
                    self.rotary_emb
                )

        # Stop gradients before final iteration
        low_level_z = low_level_z.detach()
        high_level_z = high_level_z.detach()

        # Final iteration with gradients
        low_level_z = self.low_level_reasoner(
            low_level_z,
            high_level_z + input_embeddings,
            self.rotary_emb
        )
        high_level_z = self.high_level_reasoner(
            high_level_z,
            low_level_z,
            self.rotary_emb
        )

        # Output predictions
        output_logits = self.output_head(high_level_z[:, 1:])  # Skip CLS token
        q_act_logits = self.q_act_head(high_level_z[:, 0])  # CLS token only

        return {
            'hidden_states': {
                'high_level': high_level_z.detach(),
                'low_level': low_level_z.detach()
            },
            'output': output_logits,
            'q_act_halt': q_act_logits[:, 0],
            'q_act_continue': q_act_logits[:, 1]
        }


# ============================================================================
# Sudoku Generation
# ============================================================================

class Difficulty:
    VERY_EASY = (46, 50)
    EASY = (40, 45)
    MEDIUM = (32, 39)
    HARD = (28, 31)
    EXTREME = (17, 27)


def is_valid_move(board, row, col, num):
    """Check if placing num at (row, col) is valid"""
    # Check row
    for c in range(9):
        if board[row][c] == num:
            return False
    
    # Check column
    for r in range(9):
        if board[r][col] == num:
            return False
    
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False
    
    return True


def solve_sudoku(board):
    """Simple backtracking solver"""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid_move(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def count_solutions(board, limit=2):
    """Count number of solutions (up to limit)"""
    def backtrack(board, count):
        if count[0] >= limit:
            return
        
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid_move(board, row, col, num):
                            board[row][col] = num
                            backtrack(board, count)
                            board[row][col] = 0
                    return
        count[0] += 1

    count = [0]
    board_copy = [row[:] for row in board]
    backtrack(board_copy, count)
    return count[0]


def generate_sudoku(difficulty_range):
    """Generate a Sudoku puzzle with given difficulty"""
    # Create a completed board
    board = [[0] * 9 for _ in range(9)]
    
    # Fill diagonal 3x3 boxes first (they don't affect each other)
    for box in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box + i][box + j] = nums[i * 3 + j]
    
    # Solve the rest
    solve_sudoku(board)
    solution = [row[:] for row in board]
    
    # Remove cells to create puzzle
    min_clues, max_clues = difficulty_range
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    
    puzzle = [row[:] for row in solution]
    clues = 81
    
    for row, col in cells:
        if clues <= max_clues:
            break
            
        backup = puzzle[row][col]
        puzzle[row][col] = 0
        
        if count_solutions(puzzle) != 1:
            puzzle[row][col] = backup
        else:
            clues -= 1
            if clues <= min_clues:
                break
    
    return puzzle, solution


def sudoku_board_string(board):
    """Format Sudoku board as string"""
    horizontal_line = "+-------+-------+-------+"
    result = [horizontal_line]
    
    for row_idx, row in enumerate(board):
        line = "|"
        for col_idx, cell in enumerate(row):
            display_value = "." if cell == 0 else str(cell)
            line += f" {display_value}"
            if (col_idx + 1) % 3 == 0:
                line += " |"
        result.append(line)
        
        if (row_idx + 1) % 3 == 0:
            result.append(horizontal_line)
    
    return "\n".join(result)


# ============================================================================
# Training Components
# ============================================================================

class TrainingBatch:
    DIFFICULTIES = [
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
        Difficulty.EXTREME
    ]
    
    CURRICULUM_DIFFICULTY_PROBAS = [
        [1.0, 0.0, 0.0, 0.0],  # stage 0: only easy
        [0.7, 0.3, 0.0, 0.0],  # stage 1: mostly easy, some medium
        [0.5, 0.4, 0.1, 0.0],  # stage 2: mix of easy, medium, some hard
        [0.3, 0.3, 0.3, 0.1],  # stage 3: mix of all difficulties
        [0.1, 0.3, 0.4, 0.2],  # stage 4: more hard and extreme
    ]

    def __init__(self, model, batch_size, device):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.curriculum_level = 0
        self.total_puzzles = 0
        
        # Initialize batch data
        initial_hidden = model.get_initial_hidden_states(batch_size)
        self.hidden_states = {
            'high_level': torch.zeros(batch_size, 1, model.config.transformers.hidden_size, 
                                     device=device, dtype=model.config.dtype),
            'low_level': torch.zeros(batch_size, 1, model.config.transformers.hidden_size, 
                                    device=device, dtype=model.config.dtype)
        }
        
        self.board_inputs = torch.zeros(batch_size, 81, device=device, dtype=torch.long)
        self.board_targets = torch.zeros(batch_size, 81, device=device, dtype=torch.long)
        self.segments = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        # Initialize all samples
        for i in range(batch_size):
            self.replace_sample(i)

    def sample_difficulty(self):
        """Sample difficulty based on curriculum level"""
        probabilities = self.CURRICULUM_DIFFICULTY_PROBAS[self.curriculum_level]
        rand = random.random()
        cumulative_prob = 0.0
        
        for idx, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand < cumulative_prob:
                return self.DIFFICULTIES[idx]
        
        return self.DIFFICULTIES[-1]

    def replace_sample(self, idx):
        """Replace sample at index with new puzzle"""
        # Reset hidden states
        initial_hidden = self.model.get_initial_hidden_states(1)
        self.hidden_states['high_level'][idx] = initial_hidden['high_level'][0]
        self.hidden_states['low_level'][idx] = initial_hidden['low_level'][0]
        self.segments[idx] = 0
        
        # Generate new puzzle
        difficulty = self.sample_difficulty()
        puzzle, solution = generate_sudoku(difficulty)
        
        # Convert to tensors
        puzzle_flat = [cell for row in puzzle for cell in row]
        solution_flat = [cell for row in solution for cell in row]
        
        self.board_inputs[idx] = torch.tensor(puzzle_flat, device=self.device, dtype=torch.long)
        self.board_targets[idx] = torch.tensor(solution_flat, device=self.device, dtype=torch.long)
        
        self.total_puzzles += 1

    def graduate(self):
        """Move to next curriculum level"""
        if self.curriculum_level < len(self.CURRICULUM_DIFFICULTY_PROBAS) - 1:
            self.curriculum_level += 1
            print(f"Graduated to curriculum level {self.curriculum_level}")


def sudoku_loss(model, batch, device):
    """Compute loss for Sudoku training"""
    output = model(batch.hidden_states, batch.board_inputs)
    
    # Output loss (only on unfilled squares)
    output_logits = output['output']
    output_loss = F.cross_entropy(
        output_logits.view(-1, output_logits.size(-1)),
        batch.board_targets.view(-1),
        reduction='none'
    ).view_as(batch.board_targets)
    
    # Mask for unfilled squares only
    output_loss_mask = (batch.board_inputs == 0).float()
    output_loss = (output_loss * output_loss_mask).sum() / output_loss_mask.sum()
    
    # Accuracy calculation
    predictions = output_logits.argmax(dim=-1)
    correct = (predictions == batch.board_targets) | (batch.board_inputs != 0)
    output_accuracy = correct.all(dim=1).float()
    q_act_halt_target = output_accuracy
    
    # ACT loss
    next_segments = batch.segments + 1
    is_last_segment = next_segments > model.config.act.halt_max_steps
    is_halted = is_last_segment | (output['q_act_halt'] > output['q_act_continue'])
    
    # Exploration for halting
    halt_exploration = torch.rand_like(output['q_act_halt']) < model.config.act.halt_exploration_probability
    min_halt_segments = torch.randint(2, model.config.act.halt_max_steps + 1, batch.segments.shape, device=device)
    min_halt_segments = min_halt_segments * halt_exploration.long()
    is_halted = is_halted & (next_segments > min_halt_segments)
    
    # Next step for Q-learning target
    with torch.no_grad():
        next_output = model(output['hidden_states'], batch.board_inputs)
        next_q_act_halt = next_output['q_act_halt']
        next_q_act_continue = next_output['q_act_continue']
    
    q_act_continue_target = torch.sigmoid(
        torch.where(
            is_last_segment,
            next_q_act_halt,
            torch.max(next_q_act_halt, next_q_act_continue)
        )
    )
    
    q_act_loss = (
        F.binary_cross_entropy_with_logits(output['q_act_halt'], q_act_halt_target.float()) +
        F.binary_cross_entropy_with_logits(output['q_act_continue'], q_act_continue_target)
    ) / 2
    
    # Metrics
    avg_output_full_accuracy = correct.all(dim=1).float().mean()
    avg_q_act_halt_accuracy = ((output['q_act_halt'] >= 0) == output_accuracy).float().mean()
    
    return {
        'total_loss': output_loss + q_act_loss,
        'output_loss': output_loss,
        'q_act_loss': q_act_loss,
        'is_halted': is_halted,
        'output_accuracy': avg_output_full_accuracy,
        'q_act_accuracy': avg_q_act_halt_accuracy,
        'next_hidden_states': output['hidden_states']
    }


def train_step(model, optimizer, batch, device):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    loss_dict = sudoku_loss(model, batch, device)
    loss_dict['total_loss'].backward()
    optimizer.step()
    
    # Update batch state
    batch.hidden_states = loss_dict['next_hidden_states']
    batch.segments += 1
    
    # Replace halted samples
    is_halted = loss_dict['is_halted'].cpu().numpy()
    for idx, halted in enumerate(is_halted):
        if halted:
            batch.replace_sample(idx)
    
    return {
        'output_loss': loss_dict['output_loss'].item(),
        'q_act_loss': loss_dict['q_act_loss'].item(),
        'output_accuracy': loss_dict['output_accuracy'].item(),
        'q_act_accuracy': loss_dict['q_act_accuracy'].item()
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_model():
    """Main training function"""
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = HRMACTModelConfig()
    model = HRMACTInner(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    
    # Create training batch
    batch_size = 32 if device.type == 'cpu' else 512
    batch = TrainingBatch(model, batch_size, device)
    
    # Training loop
    step_idx = 0
    steps_since_graduation = 0
    accuracy_history = []
    
    print("Starting training...")
    
    while True:
        step_idx += 1
        steps_since_graduation += 1
        
        # Training step
        metrics = train_step(model, optimizer, batch, device)
        
        # Print progress
        if step_idx % 10 == 0:
            print(f"Step {step_idx} | "
                  f"Output Loss: {metrics['output_loss']:.4f} | "
                  f"Output Acc: {metrics['output_accuracy']:.4f} | "
                  f"Q-ACT Loss: {metrics['q_act_loss']:.4f} | "
                  f"Q-ACT Acc: {metrics['q_act_accuracy']:.4f} | "
                  f"Puzzles: {batch.total_puzzles} | "
                  f"Curriculum: {batch.curriculum_level}")
        
        # Save checkpoints
        if step_idx == 1 or step_idx % 250 == 0:
            checkpoint_path = f"checkpoint-{step_idx}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step_idx,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Curriculum progression
        accuracy_history.append(metrics['output_accuracy'])
        if len(accuracy_history) > 300:
            accuracy_history.pop(0)
        
        if len(accuracy_history) >= 300:
            avg_rolling_accuracy = sum(accuracy_history) / len(accuracy_history)
            if avg_rolling_accuracy >= 0.85 and steps_since_graduation >= 300:
                steps_since_graduation = 0
                batch.graduate()


# ============================================================================
# Inference
# ============================================================================

def infer_model(checkpoint_path, difficulty_range=Difficulty.MEDIUM):
    """Run inference on a Sudoku puzzle"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = HRMACTInner(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Loaded model!")
    
    # Generate puzzle
    puzzle, solution = generate_sudoku(difficulty_range)
    print(f"Puzzle:\n{sudoku_board_string(puzzle)}")
    print(f"Solution:\n{sudoku_board_string(solution)}")
    
    # Convert to tensor
    puzzle_flat = torch.tensor([cell for row in puzzle for cell in row], 
                              device=device, dtype=torch.long).unsqueeze(0)
    solution_flat = [cell for row in solution for cell in row]
    
    # Initialize hidden states
    hidden_states = model.get_initial_hidden_states(1)
    
    with torch.no_grad():
        for segment in range(1, config.act.halt_max_steps + 1):
            print(f"\nSegment {segment}")
            
            output = model(hidden_states, puzzle_flat)
            hidden_states = output['hidden_states']
            
            predictions = output['output'][0].argmax(dim=-1).cpu().numpy()
            
            # Calculate accuracy
            accurate_squares = 0
            predicted_squares = 0
            predicted_flat_board = []
            
            for i, (puzzle_cell, solution_cell, predicted_cell) in enumerate(
                zip([cell for row in puzzle for cell in row], solution_flat, predictions)
            ):
                if puzzle_cell != 0:
                    predicted_flat_board.append(puzzle_cell)
                else:
                    predicted_flat_board.append(predicted_cell)
                    if predicted_cell == solution_cell:
                        accurate_squares += 1
                    predicted_squares += 1
            
            # Convert back to 9x9 board
            predicted_board = []
            for i in range(0, 81, 9):
                predicted_board.append(predicted_flat_board[i:i+9])
            
            print(f"Predicted solution ({accurate_squares} / {predicted_squares}):")
            print(sudoku_board_string(predicted_board))
            
            # Check halting decision
            q_halt = torch.sigmoid(output['q_act_halt'][0]).item()
            q_continue = torch.sigmoid(output['q_act_continue'][0]).item()
            print(f"Q (halt - continue): {q_halt:.3f} - {q_continue:.3f}")
            
            if q_halt > q_continue:
                print("Halting.")
                break


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HRM Sudoku Solver')
    parser.add_argument('mode', choices=['train', 'infer'], help='Mode to run')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for inference')
    parser.add_argument('--difficulty', type=str, choices=['very-easy', 'easy', 'medium', 'hard', 'extreme'],
                       default='medium', help='Difficulty for inference')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'infer':
        if not args.checkpoint:
            print("Error: --checkpoint required for inference mode")
            return
        
        # Map difficulty string to range
        difficulty_map = {
            'very-easy': Difficulty.VERY_EASY,
            'easy': Difficulty.EASY,
            'medium': Difficulty.MEDIUM,
            'hard': Difficulty.HARD,
            'extreme': Difficulty.EXTREME
        }
        
        difficulty_range = difficulty_map[args.difficulty]
        infer_model(args.checkpoint, difficulty_range)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()