import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import warnings
import time
from collections import deque
from datasets import load_dataset

from config import *
from model import build_llama
from dataset import StreamingLanguageModelDataset

# ==============================================================================
# CONFIGURATION CONSTANTS (STRICT CURRICULUM)
# ==============================================================================
TOKEN_CAP_PER_PHASE = 500_000  # Reduced for verification

# Progressive LR Decay
LR_PHASE_1 = 3e-4  # High (Bootstrapping)
LR_PHASE_2 = 1e-4  # Medium (Structure)
LR_PHASE_3 = 5e-5  # Low (Generalization)

def get_model(vocab_size):
    return build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )

def train_phase(model, optimizer, scaler, dataset_name, phase_name, num_epochs, target_lr, vocab_size):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Update Learning Rate for this phase
    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr
        
    print("\n" + "=" * 80)
    print(f"STARTING PHASE: {phase_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {num_epochs} (Logical Passes)")
    print(f"Learning Rate: {target_lr}")
    print(f"Token Cap: {TOKEN_CAP_PER_PHASE:,}")
    print("=" * 80)

    total_phase_tokens = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} of {phase_name} ---")
        
        # Load Dataset Stream (Restarted each epoch)
        try:
            # Handle potential dataset loading errors (e.g. bookcorpus vs bookcorpus/bookcorpus)
            if dataset_name == "bookcorpus":
                ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return

        train_dataset = StreamingLanguageModelDataset(
            ds,
            seq_len=SEQ_LEN,
            tokenizer_name="cl100k_base"
        )
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            pin_memory=True
        )

        model.train()
        loss_window = deque(maxlen=50)
        pbar = tqdm(dataloader, desc=f"{phase_name} E{epoch+1}", dynamic_ncols=True)
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            
            # --- TOKEN ACCOUNTING ---
            batch_tokens = input_ids.numel()
            if total_phase_tokens + batch_tokens > TOKEN_CAP_PER_PHASE:
                print(f"\n[STOP] Token Cap Reached for {phase_name}: {total_phase_tokens + batch_tokens:,} > {TOKEN_CAP_PER_PHASE:,}")
                return  # End Phase Immediately
            
            total_phase_tokens += batch_tokens

            # --- OPTIMIZATION STEP ---
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(input_ids)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # --- STATS ---
            loss_window.append(loss.item())
            avg_loss = sum(loss_window) / len(loss_window)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{target_lr:.1e}",
                "Tok": f"{total_phase_tokens/1e6:.2f}M"
            })
            
        print(f"Epoch {epoch+1} Complete. Tokens so far: {total_phase_tokens:,}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    
    vocab_size = 100277
    model = get_model(vocab_size).to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Initialize Optimizer (Maintains state across phases)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_PHASE_1,  # Start with Phase 1 LR
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")

    # ==========================================================================
    # PHASE 1: TinyStories (Bootstrapping) - 2 Epochs, High LR
    # ==========================================================================
    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="roneneldan/TinyStories",
        phase_name="Phase 1 (TinyStories)",
        num_epochs=2,
        target_lr=LR_PHASE_1,
        vocab_size=vocab_size
    )
    
    # Save Checkpoint after Phase 1
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase1.pt")

    # ==========================================================================
    # PHASE 2: BookCorpus (Structure) - 2 Epochs, Medium LR
    # ==========================================================================
    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="bookcorpus", 
        phase_name="Phase 2 (BookCorpus)",
        num_epochs=2,
        target_lr=LR_PHASE_2,
        vocab_size=vocab_size
    )
    
    # Save Checkpoint after Phase 2
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase2.pt")

    # ==========================================================================
    # PHASE 3: OpenWebText (Generalization) - 1 Epoch, Low LR, Truncated to Cap
    # ==========================================================================
    train_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset_name="Skylion007/openwebtext",
        phase_name="Phase 3 (OpenWebText)",
        num_epochs=1,
        target_lr=LR_PHASE_3,
        vocab_size=vocab_size
    )
    
    # Final Save
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/model_final.pt")
    print("\nTRAINING COMPLETE.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
