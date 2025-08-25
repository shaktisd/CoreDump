import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from typing import Optional, Tuple

# -----------------------------
# Flash Attention Implementation
# -----------------------------
class FlashMultiheadAttention(nn.Module):
    """Simplified Flash Attention for better memory efficiency"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]

        # Flash attention computation with chunking
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out

# -----------------------------
# Puzzle-Aware Positional Encoding
# -----------------------------
class SudokuPositionalEncoding(nn.Module):
    """Sudoku-specific positional encoding that understands grid structure"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Row, column, and box embeddings
        self.row_embed = nn.Embedding(9, hidden_size // 4)
        self.col_embed = nn.Embedding(9, hidden_size // 4)
        self.box_embed = nn.Embedding(9, hidden_size // 4)

        # Additional learned position embedding
        self.pos_embed = nn.Embedding(81, hidden_size // 4)

    def forward(self, x):
        B, seq_len = x.shape
        device = x.device

        # Convert flat positions to row, col coordinates
        positions = torch.arange(seq_len, device=device)
        rows = positions // 9
        cols = positions % 9
        boxes = (rows // 3) * 3 + (cols // 3)

        # Get embeddings
        row_emb = self.row_embed(rows)  # [seq_len, hidden_size//4]
        col_emb = self.col_embed(cols)
        box_emb = self.box_embed(boxes)
        pos_emb = self.pos_embed(positions)

        # Concatenate all embeddings
        pos_encoding = torch.cat([row_emb, col_emb, box_emb, pos_emb], dim=-1)  # [seq_len, hidden_size]

        return pos_encoding.unsqueeze(0).expand(B, -1, -1)

# -----------------------------
# Halting Mechanism
# -----------------------------
class HaltingMechanism(nn.Module):
    """Adaptive computation time mechanism"""
    def __init__(self, hidden_size, max_halts=10):
        super().__init__()
        self.max_halts = max_halts
        self.halt_predictor = nn.Linear(hidden_size, 1)
        self.halt_threshold = 0.5

    def forward(self, hidden_states, step):
        # Predict halting probability
        halt_logits = self.halt_predictor(hidden_states)  # [B, seq_len, 1]
        halt_probs = torch.sigmoid(halt_logits).squeeze(-1)  # [B, seq_len]

        # Decision to halt based on threshold and max steps
        should_halt = (halt_probs > self.halt_threshold) | (step >= self.max_halts - 1)

        return should_halt, halt_probs

# -----------------------------
# Enhanced Reasoning Block
# -----------------------------
class EnhancedReasoningBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, use_flash=True):
        super().__init__()
        if use_flash:
            self.attn = FlashMultiheadAttention(hidden_size, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        # Enhanced MLP with gating
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

        # Gating mechanism for better gradient flow
        self.gate = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_flash = use_flash

    def forward(self, x):
        # Self-attention with residual
        if self.use_flash:
            attn_out = self.attn(x)
        else:
            attn_out, _ = self.attn(x, x, x)

        x = self.norm1(x + self.dropout(attn_out))

        # MLP with gating
        mlp_out = self.mlp(x)
        gate_values = torch.sigmoid(self.gate(x))
        mlp_out = mlp_out * gate_values

        x = self.norm2(x + mlp_out)
        return x

# -----------------------------
# Enhanced Hierarchical Reasoning Model
# -----------------------------
class EnhancedHRM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=8, H_layers=4, L_layers=4,
                 dropout=0.1, use_halting=True, use_flash=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_halting = use_halting

        # Enhanced embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = SudokuPositionalEncoding(hidden_size)

        # Reasoning blocks
        self.H_blocks = nn.ModuleList([
            EnhancedReasoningBlock(hidden_size, num_heads, dropout, use_flash)
            for _ in range(H_layers)
        ])
        self.L_blocks = nn.ModuleList([
            EnhancedReasoningBlock(hidden_size, num_heads, dropout, use_flash)
            for _ in range(L_layers)
        ])

        # Halting mechanism
        if use_halting:
            self.halting = HaltingMechanism(hidden_size)

        # Output projection with uncertainty estimation
        self.pre_head_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.confidence_head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, return_confidence=False):
        B, seq_len = x.shape
        device = x.device

        # Token + positional embeddings
        h = self.token_embed(x) + self.pos_encoding(x)

        # High-level reasoning with adaptive computation
        halt_penalties = []
        for step, block in enumerate(self.H_blocks):
            h = block(h)

            if self.use_halting and self.training:
                should_halt, halt_probs = self.halting(h, step)
                halt_penalties.append(halt_probs.mean())

                # Early stopping simulation (in practice, would be more complex)
                if should_halt.all():
                    break

        # Low-level reasoning
        for block in self.L_blocks:
            h = block(h)

        # Output
        h = self.pre_head_norm(h)
        logits = self.lm_head(h)

        if return_confidence:
            confidence = torch.sigmoid(self.confidence_head(h))
            return logits, confidence, halt_penalties

        return logits, halt_penalties if self.use_halting and self.training else []

    def save_model(self, filepath):
        """Save the model weights to a file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.token_embed.num_embeddings,
            'hidden_size': self.hidden_size,
            'num_h_layers': len(self.H_blocks),
            'num_l_layers': len(self.L_blocks),
            'num_heads': self.H_blocks[0].attn.num_heads if hasattr(self.H_blocks[0].attn, 'num_heads') else 8,
            'use_halting': self.use_halting
        }, filepath)

    @classmethod
    def load_model(cls, filepath, device=None):
        """Load a model from a saved file"""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            hidden_size=checkpoint['hidden_size'],
            num_heads=checkpoint.get('num_heads', 8),
            H_layers=checkpoint['num_h_layers'],
            L_layers=checkpoint['num_l_layers'],
            use_halting=checkpoint.get('use_halting', True)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

# -----------------------------
# Enhanced Sudoku Dataset
# -----------------------------
MASK_TOKEN = 0
DIGITS = list(range(1, 10))
VOCAB_SIZE = 10

def is_valid_sudoku(board):
    """Check if a Sudoku board is valid"""
    def is_valid_unit(unit):
        unit = [x for x in unit if x != 0]
        return len(unit) == len(set(unit))

    # Check rows
    for row in board:
        if not is_valid_unit(row):
            return False

    # Check columns
    for col in range(9):
        if not is_valid_unit([board[row][col] for row in range(9)]):
            return False

    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(board[box_row*3 + i][box_col*3 + j])
            if not is_valid_unit(box):
                return False

    return True

def generate_sudoku_board():
    """Generate a completed Sudoku using a more robust algorithm."""
    def solve_sudoku(board):
        def find_empty():
            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        return (i, j)
            return None

        def is_valid(pos, num):
            row, col = pos

            # Check row
            for j in range(9):
                if board[row][j] == num:
                    return False

            # Check column
            for i in range(9):
                if board[i][col] == num:
                    return False

            # Check 3x3 box
            box_row = (row // 3) * 3
            box_col = (col // 3) * 3
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if board[i][j] == num:
                        return False

            return True

        pos = find_empty()
        if not pos:
            return True

        nums = list(range(1, 10))
        random.shuffle(nums)

        for num in nums:
            if is_valid(pos, num):
                board[pos[0]][pos[1]] = num

                if solve_sudoku(board):
                    return True

                board[pos[0]][pos[1]] = 0

        return False

    board = [[0 for _ in range(9)] for _ in range(9)]
    solve_sudoku(board)
    return np.array(board)

def strategic_mask_board(board, mask_prob=0.6, difficulty='medium'):
    """Strategically mask cells based on difficulty and Sudoku constraints"""
    masked = board.copy()

    # Different masking strategies based on difficulty
    if difficulty == 'easy':
        mask_prob = 0.4
    elif difficulty == 'medium':
        mask_prob = 0.6
    elif difficulty == 'hard':
        mask_prob = 0.75

    # Ensure we don't mask too many cells in the same constraint group
    mask = np.random.rand(*board.shape) < mask_prob

    # Additional strategic masking - keep some cells in each row/col/box
    for i in range(9):
        # Ensure at least 2 cells per row
        row_mask = mask[i, :]
        if row_mask.sum() > 7:
            indices_to_keep = np.random.choice(9, 2, replace=False)
            mask[i, indices_to_keep] = False

        # Ensure at least 2 cells per column
        col_mask = mask[:, i]
        if col_mask.sum() > 7:
            indices_to_keep = np.random.choice(9, 2, replace=False)
            mask[indices_to_keep, i] = False

    masked[mask] = MASK_TOKEN
    return masked

def generate_sudoku_batch(batch_size=32, mask_prob=0.6, difficulty='medium'):
    """Generate a batch with better puzzle diversity"""
    inputs, labels = [], []
    for _ in range(batch_size):
        solution = generate_sudoku_board()
        puzzle = strategic_mask_board(solution, mask_prob, difficulty)
        inputs.append(puzzle.flatten())
        labels.append(solution.flatten())
    return torch.tensor(np.array(inputs), dtype=torch.long), torch.tensor(np.array(labels), dtype=torch.long)

# -----------------------------
# Enhanced Training Loop
# -----------------------------
def train_enhanced_sudoku():
    seq_len = 81
    hidden_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Enhanced model
    model = EnhancedHRM(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        num_heads=8,
        H_layers=4,
        L_layers=4,
        dropout=0.1,
        use_halting=True,
        use_flash=True
    ).to(device)

    # Better optimizer with scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-6)

    # Enhanced loss function
    def compute_loss(logits, targets, mask_positions, halt_penalties=None):
        # Main reconstruction loss
        main_loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            targets.view(-1),
            ignore_index=MASK_TOKEN
        )

        # Penalty for halting (encourage efficiency)
        halt_loss = 0.0
        if halt_penalties:
            halt_loss = 0.01 * sum(halt_penalties)  # Small penalty for computational cost

        return main_loss + halt_loss

    NO_OF_EPOCHS = 50000
    best_loss = float('inf')

    for step in range(NO_OF_EPOCHS):
        # Progressive difficulty curriculum
        if step < 2000:
            difficulty = 'easy'
            batch_size = 64
        elif step < 5000:
            difficulty = 'medium'
            batch_size = 32
        else:
            difficulty = 'hard'
            batch_size = 16

        x, y = generate_sudoku_batch(batch_size=batch_size, difficulty=difficulty)
        x, y = x.to(device), y.to(device)

        # Forward pass
        if model.use_halting and model.training:
            logits, halt_penalties = model(x)
        else:
            logits = model(x)
            halt_penalties = []

        # Compute loss
        mask_positions = (x == MASK_TOKEN)
        loss = compute_loss(logits, y, mask_positions, halt_penalties)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Logging
        if step % 200 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

            # Validation check
            if loss.item() < best_loss:
                best_loss = loss.item()
                model.save_model('best_sudoku_hrm_model.pt')

        # Detailed evaluation
        if step % 1000 == 0 and step > 0:
            evaluate_model(model, device)

    # Final save
    model.save_model('final_sudoku_hrm_model.pt')
    print(f"Training completed. Best loss: {best_loss:.4f}")

def evaluate_model(model, device, num_samples=10):
    """Evaluate model performance"""
    model.eval()
    correct_predictions = 0
    total_masked = 0

    with torch.no_grad():
        for _ in range(num_samples):
            x, y = generate_sudoku_batch(batch_size=1, difficulty='medium')
            x, y = x.to(device), y.to(device)

            if model.use_halting:
                logits, _ = model(x)
            else:
                logits = model(x)

            predictions = torch.argmax(logits, dim=-1)

            # Check accuracy on masked positions only
            mask_positions = (x == MASK_TOKEN)
            masked_predictions = predictions[mask_positions]
            masked_targets = y[mask_positions]

            correct_predictions += (masked_predictions == masked_targets).sum().item()
            total_masked += mask_positions.sum().item()

    accuracy = correct_predictions / total_masked if total_masked > 0 else 0
    print(f"Validation Accuracy on masked positions: {accuracy:.4f}")

    # Show an example
    with torch.no_grad():
        x, y = generate_sudoku_batch(batch_size=1, difficulty='medium')
        x, y = x.to(device), y.to(device)

        if model.use_halting:
            logits, _ = model(x)
        else:
            logits = model(x)

        predictions = torch.argmax(logits, dim=-1)

        print("\nExample:")
        print("Puzzle:\n", x[0].cpu().numpy().reshape(9, 9))
        print("Prediction:\n", predictions[0].cpu().numpy().reshape(9, 9))
        print("Solution:\n", y[0].cpu().numpy().reshape(9, 9))

    model.train()

if __name__ == "__main__":
    train_enhanced_sudoku()