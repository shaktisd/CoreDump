import torch
import numpy as np
from enhanced_simple_hrm_model import EnhancedHRM, generate_sudoku_batch, is_valid_sudoku
from tqdm import tqdm
import time

def evaluate_model_performance(model_path, num_samples=1000, difficulty='medium'):
    """
    Evaluate model performance on a large number of Sudoku puzzles
    
    Args:
        model_path: Path to the saved model checkpoint
        num_samples: Number of puzzles to evaluate
        difficulty: Difficulty level of puzzles ('easy', 'medium', 'hard')
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and move to device
    print(f"Loading model from {model_path}")
    model = EnhancedHRM.load_model(model_path)
    model = model.to(device)
    
    # Ensure all model components are on the correct device
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight = torch.nn.Parameter(module.weight.to(device))
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias = torch.nn.Parameter(module.bias.to(device))
    
    model.eval()

    # Metrics
    total_masked = 0
    correct_predictions = 0
    valid_solutions = 0
    total_time = 0
    confidence_scores = []

    print(f"\nEvaluating model on {num_samples} {difficulty} difficulty puzzles...")
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            # Generate puzzle and ensure it's on the correct device
            x, y = generate_sudoku_batch(batch_size=1, difficulty=difficulty)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Time the inference
            start_time = time.time()
            try:
                outputs = model(x, return_confidence=True)
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        logits, confidence, _ = outputs
                    else:
                        logits, confidence = outputs
                else:
                    logits = outputs
                    # Create default confidence tensor with matching batch size
                    confidence = torch.ones(x.shape, device=device)  # default confidence if not provided
            except Exception as e:
                print(f"Error during inference: {e}")
                continue
            end_time = time.time()
            total_time += (end_time - start_time)

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy on masked positions
            mask_positions = (x == 0)  # 0 is MASK_TOKEN
            masked_predictions = predictions[mask_positions]
            masked_targets = y[mask_positions]
            
            correct_predictions += (masked_predictions == masked_targets).sum().item()
            total_masked += mask_positions.sum().item()

            # Check if solution is valid Sudoku
            predicted_grid = predictions[0].cpu().numpy().reshape(9, 9)
            if is_valid_sudoku(predicted_grid):
                valid_solutions += 1

            # Reshape confidence tensor if needed and calculate average confidence score
            if confidence.dim() == 3:
                confidence = confidence.squeeze(-1)  # Remove last dimension if it exists
            confidence = confidence.view(x.shape)  # Reshape to match input shape
            
            # Calculate average confidence for masked positions
            masked_confidence = confidence[mask_positions]
            avg_confidence = masked_confidence.mean().item()
            confidence_scores.append(avg_confidence)

            # Print detailed analysis for first few puzzles
            if i < 3:
                print(f"\nDetailed analysis for puzzle {i+1}:")
                input_grid = x[0].cpu().numpy().reshape(9, 9)
                true_grid = y[0].cpu().numpy().reshape(9, 9)
                
                print("\nInterleaved Row-by-Row Analysis:")
                for row in range(9):
                    print(f"\nRow {row+1}:")
                    print("-" * 35)
                    # Format each number with proper spacing
                    input_row = " ".join(f"{n:2d}" for n in input_grid[row])
                    model_row = " ".join(f"{n:2d}" for n in predicted_grid[row])
                    true_row = " ".join(f"{n:2d}" for n in true_grid[row])
                    print(f"Input:  {input_row}")
                    print(f"Model:  {model_row}")
                    print(f"True:   {true_row}")
                    print("-" * 35)
                print(f"\nAverage confidence score: {avg_confidence:.4f}")
                print(f"Valid solution: {is_valid_sudoku(predicted_grid)}")

    # Calculate final metrics
    accuracy = correct_predictions / total_masked if total_masked > 0 else 0
    valid_percentage = (valid_solutions / num_samples) * 100
    avg_time_per_puzzle = total_time / num_samples
    avg_confidence = np.mean(confidence_scores)
    confidence_std = np.std(confidence_scores)

    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Number of puzzles evaluated: {num_samples}")
    print(f"Difficulty level: {difficulty}")
    print(f"Accuracy on masked positions: {accuracy:.4f}")
    print(f"Percentage of valid solutions: {valid_percentage:.2f}%")
    print(f"Average inference time per puzzle: {avg_time_per_puzzle*1000:.2f}ms")
    print(f"Average confidence score: {avg_confidence:.4f} ± {confidence_std:.4f}")
    print(f"Total masked positions evaluated: {total_masked}")
    print(f"Total correct predictions: {correct_predictions}")
    print("=" * 50)

if __name__ == "__main__":
    # You can change the model path and number of samples here
    model_path = r"C:\00_project_code\micro-llm-project\final_sudoku_hrm_model.pt"  # or "final_sudoku_hrm_model.pt"
    evaluate_model_performance(
        model_path=model_path,
        num_samples=100,
        difficulty='medium'
    )
