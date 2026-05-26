"""
================================================================
  DEEPFAKE AUDIO DETECTION — COMBINED main.py
================================================================
  Usage:
    python main.py --train       # Train the model (first time)
    python main.py --detect      # Run live mic detection
    python main.py               # Auto: train if no model, else detect

  Mathematical Framework:
    1. AAS  = α·P_real − β·P_fake − γ·H(p)       [Authenticity Score]
    2. SDI  = mean|features − μ_real|              [Spectral Deviation]
    3. TCS  = 1 − σ_input / (σ_real + ε)          [Temporal Consistency]
    4. FDI  = w1·AAS + w2·(1−SDI) + w3·TCS        [Final Fraud Index]
    5. Loss = BCE + λ·MSE(FDI, label)              [Custom Loss]

  SIR Classification:
    🟢 S — Genuine    (FDI ≥ 0.70)
    🟡 I — Suspicious (0.40 ≤ FDI < 0.70)
    🔴 R — Deepfake   (FDI < 0.40)

  Install:
    pip install datasets transformers torch torchaudio sounddevice
                soundfile scikit-learn numpy matplotlib
================================================================
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME     = "facebook/wav2vec2-base"
MODEL_PATH     = "./deepfake_model"
SAMPLE_RATE    = 16000
RECORD_SECONDS = 5

NUM_REAL   = 400
NUM_FAKE   = 400
EPOCHS     = 3
BATCH_SIZE = 4
LR         = 3e-5

# Formula weights — AAS
ALPHA = 0.5
BETA  = 0.3
GAMMA = 0.2

# Formula weights — FDI
W1 = 0.50
W2 = 0.30
W3 = 0.20

# Loss balance
LAMBDA = 0.4

# SIR thresholds
SIR_S = 0.70
SIR_I = 0.40
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# FORMULA 1 — Audio Authenticity Score (AAS)
# AAS = α·P_real − β·P_fake − γ·H(p)
# ══════════════════════════════════════════════
def compute_AAS(p_real, p_fake):
    p = torch.stack([p_real, p_fake], dim=-1).clamp(1e-8, 1.0)
    entropy = -torch.sum(p * torch.log(p), dim=-1)
    return ALPHA * p_real - BETA * p_fake - GAMMA * entropy


# ══════════════════════════════════════════════
# FORMULA 2 — Spectral Deviation Index (SDI)
# SDI = (1/N) Σ |feature_input − μ_real|
# ══════════════════════════════════════════════
def compute_SDI(features, real_mean):
    feat_len = features.shape[-1]
    if real_mean.shape[0] >= feat_len:
        mean_trim = real_mean[:feat_len]
    else:
        mean_trim = F.pad(real_mean, (0, feat_len - real_mean.shape[0]))
    deviation = torch.abs(features - mean_trim.unsqueeze(0))
    return torch.sigmoid(deviation.mean(dim=-1))


# ══════════════════════════════════════════════
# FORMULA 3 — Temporal Consistency Score (TCS)
# TCS = 1 − σ_input / (σ_real + ε)
# ══════════════════════════════════════════════
def compute_TCS(features, real_std, eps=1e-6):
    input_std = features.std(dim=-1)
    tcs = 1.0 - (input_std / (real_std.mean() + eps))
    return torch.clamp(tcs, 0.0, 1.0)


# ══════════════════════════════════════════════
# FORMULA 4 — Fraud Detection Index (FDI)
# FDI = w1·AAS + w2·(1−SDI) + w3·TCS
# ══════════════════════════════════════════════
def compute_FDI(aas, sdi, tcs):
    aas_norm = torch.sigmoid(aas)
    return W1 * aas_norm + W2 * (1 - sdi) + W3 * tcs


# ══════════════════════════════════════════════
# FORMULA 5 — Custom Combined Loss
# Loss = BCE(logits, label) + λ·MSE(FDI, target)
# ══════════════════════════════════════════════
class CustomAudioLoss(nn.Module):
    def __init__(self, lambda_weight=LAMBDA):
        super().__init__()
        self.lambda_w = lambda_weight
        self.bce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, logits, fdi, labels):
        bce_loss   = self.bce(logits, labels)
        fdi_target = 1.0 - labels.float()   # real→1.0, fake→0.0
        fdi_loss   = self.mse(fdi, fdi_target)
        total      = bce_loss + self.lambda_w * fdi_loss
        return total, bce_loss, fdi_loss


# ══════════════════════════════════════════════
# SIR CLASSIFICATION
# ══════════════════════════════════════════════
def classify_SIR(fdi_score):
    if fdi_score >= SIR_S:
        return "S", "🟢 S — Genuine Audio (Safe)"
    elif fdi_score >= SIR_I:
        return "I", "🟡 I — Suspicious Audio (Needs Review)"
    else:
        return "R", "🔴 R — Confirmed Deepfake (ALERT)"


# ══════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════
def train():
    from datasets import load_dataset, Audio, concatenate_datasets
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print("  DEEPFAKE AUDIO DETECTION — TRAINING")
    print(f"{'='*55}")
    print(f"  Device : {device}")

    # ── 1. Load dataset ──────────────────────────────────────
    print("\n[1/6] Loading dataset from HuggingFace...")
    print("      (First run downloads ~1.16 GB — please wait)\n")
    dataset    = load_dataset("garystafford/deepfake-audio-detection")
    dataset    = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    train_data = dataset["train"]

    real     = train_data.filter(lambda x: x["label"] == 0).select(range(NUM_REAL))
    fake     = train_data.filter(lambda x: x["label"] == 1).select(range(NUM_FAKE))
    combined = concatenate_datasets([real, fake]).shuffle(seed=42)
    split    = combined.train_test_split(test_size=0.2, seed=42)
    print(f"      Train: {len(split['train'])} | Val: {len(split['test'])}")

    # ── 2. Load pretrained model ──────────────────────────────
    print("\n[2/6] Loading pretrained wav2vec2 model...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    # ── 3. Preprocess ─────────────────────────────────────────
    print("\n[3/6] Preprocessing audio features...")

    def preprocess(batch):
        arrays = [x["array"] for x in batch["audio"]]
        inputs = feature_extractor(
            arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding=True, max_length=SAMPLE_RATE * 5, truncation=True,
        )
        inputs["labels"] = batch["label"]
        return inputs

    train_set = split["train"].map(preprocess, batched=True, batch_size=8,
                                   remove_columns=["audio"])
    val_set   = split["test"].map(preprocess,  batched=True, batch_size=8,
                                   remove_columns=["audio"])
    train_set.set_format("torch")
    val_set.set_format("torch")

    # ── 4. Real-voice statistics ──────────────────────────────
    print("\n[4/6] Computing real-voice spectral statistics (μ, σ)...")
    real_features = [item["input_values"].float()
                     for item in train_set if item["labels"].item() == 0]
    max_len     = max(t.shape[0] for t in real_features)
    real_padded = [F.pad(t, (0, max_len - t.shape[0])) for t in real_features]
    real_tensor = torch.stack(real_padded).to(device)
    real_mean   = real_tensor.mean(dim=0)
    real_std    = real_tensor.std(dim=0).clamp(min=1e-6)

    # ── 5. Training loop ──────────────────────────────────────
    loss_fn   = CustomAudioLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    history   = {"loss": [], "bce": [], "fdi_loss": [], "val_acc": []}

    print(f"\n[5/6] Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = epoch_bce = epoch_fdi = 0.0
        n_batches  = 0
        indices    = np.random.permutation(len(train_set)).tolist()

        for start in range(0, len(indices), BATCH_SIZE):
            batch  = train_set[indices[start:start + BATCH_SIZE]]
            iv_list = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                       for v in batch["input_values"]]
            bmax    = max(t.shape[0] for t in iv_list)
            iv_pad  = [F.pad(t, (0, bmax - t.shape[0])) for t in iv_list]
            input_values = torch.stack(iv_pad).to(device)
            labels       = torch.tensor(batch["labels"]).to(device)

            optimizer.zero_grad()
            outputs = model(input_values=input_values)
            logits  = outputs.logits
            probs   = F.softmax(logits, dim=-1)

            aas  = compute_AAS(probs[:, 0], probs[:, 1])
            sdi  = compute_SDI(input_values, real_mean)
            tcs  = compute_TCS(input_values, real_std)
            fdi  = compute_FDI(aas, sdi, tcs)

            loss, bce_l, fdi_l = loss_fn(logits, fdi, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce  += bce_l.item()
            epoch_fdi  += fdi_l.item()
            n_batches  += 1

        # Validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for start in range(0, len(val_set), BATCH_SIZE):
                batch   = val_set[start:start + BATCH_SIZE]
                iv_list = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                           for v in batch["input_values"]]
                bmax    = max(t.shape[0] for t in iv_list)
                iv      = torch.stack([F.pad(t, (0, bmax - t.shape[0])) for t in iv_list]).to(device)
                lbl     = torch.tensor(batch["labels"]).to(device)
                out     = model(input_values=iv)
                correct += (out.logits.argmax(-1) == lbl).sum().item()
                total   += len(lbl)

        val_acc = correct / total
        history["loss"].append(epoch_loss / n_batches)
        history["bce"].append(epoch_bce / n_batches)
        history["fdi_loss"].append(epoch_fdi / n_batches)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {epoch_loss/n_batches:.4f} | "
              f"BCE: {epoch_bce/n_batches:.4f} | "
              f"FDI Loss: {epoch_fdi/n_batches:.4f} | "
              f"Val Acc: {val_acc*100:.1f}%")

    # ── 6. Save model + stats ─────────────────────────────────
    print(f"\n[6/6] Saving model to '{MODEL_PATH}'...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    feature_extractor.save_pretrained(MODEL_PATH)
    np.save(f"{MODEL_PATH}/real_mean.npy", real_mean.cpu().numpy())
    np.save(f"{MODEL_PATH}/real_std.npy",  real_std.cpu().numpy())

    # Final classification report
    all_preds, all_labels, all_fdi = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_set), BATCH_SIZE):
            batch   = val_set[start:start + BATCH_SIZE]
            iv_list = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                       for v in batch["input_values"]]
            bmax    = max(t.shape[0] for t in iv_list)
            iv      = torch.stack([F.pad(t, (0, bmax - t.shape[0])) for t in iv_list]).to(device)
            lbl     = torch.tensor(batch["labels"]).to(device)
            out     = model(input_values=iv)
            probs   = F.softmax(out.logits, dim=-1)
            aas     = compute_AAS(probs[:, 0], probs[:, 1])
            sdi     = compute_SDI(iv, real_mean)
            tcs     = compute_TCS(iv, real_std)
            fdi     = compute_FDI(aas, sdi, tcs)
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            all_fdi.extend(fdi.cpu().numpy())

    print("\n" + "="*55)
    print("  FINAL EVALUATION REPORT")
    print("="*55)
    print(classification_report(all_labels, all_preds,
                                 target_names=["REAL", "DEEPFAKE"]))

    _plot_results(history, all_labels, all_fdi)
    print(f"\n✅ Model saved to '{MODEL_PATH}'. Run with --detect to start!\n")


def _plot_results(history, labels, fdi_scores):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Deepfake Audio Detection — Training Results", fontsize=14)

    axes[0, 0].plot(history["loss"],     label="Total Loss")
    axes[0, 0].plot(history["bce"],      label="BCE Loss")
    axes[0, 0].plot(history["fdi_loss"], label="FDI Loss (λ·MSE)")
    axes[0, 0].set_title("Training Loss Curve")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history["val_acc"], color="green", marker="o")
    axes[0, 1].set_title("Validation Accuracy per Epoch")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True)

    real_fdi = [fdi_scores[i] for i in range(len(labels)) if labels[i] == 0]
    fake_fdi = [fdi_scores[i] for i in range(len(labels)) if labels[i] == 1]
    axes[1, 0].hist(real_fdi, bins=20, alpha=0.7, label="Real",     color="green")
    axes[1, 0].hist(fake_fdi, bins=20, alpha=0.7, label="Deepfake", color="red")
    axes[1, 0].axvline(SIR_S, color="blue",   linestyle="--", label=f"S={SIR_S}")
    axes[1, 0].axvline(SIR_I, color="orange", linestyle="--", label=f"I={SIR_I}")
    axes[1, 0].set_title("FDI Score Distribution (S→I→R)")
    axes[1, 0].set_xlabel("FDI Score"); axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend(); axes[1, 0].grid(True)

    counts = {"S\n(Genuine)": 0, "I\n(Suspicious)": 0, "R\n(Deepfake)": 0}
    for s in fdi_scores:
        if   s >= SIR_S: counts["S\n(Genuine)"]   += 1
        elif s >= SIR_I: counts["I\n(Suspicious)"] += 1
        else:            counts["R\n(Deepfake)"]    += 1
    axes[1, 1].bar(counts.keys(), counts.values(),
                   color=["green", "orange", "red"])
    axes[1, 1].set_title("SIR Stage Distribution")
    axes[1, 1].set_xlabel("Stage"); axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("      📊 Saved: training_results.png")


# ══════════════════════════════════════════════
# DETECTION (INFERENCE)
# ══════════════════════════════════════════════
def _load_model():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print("\n[1/3] Loading saved model (offline — no internet needed)...")
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  ERROR: No model found at '{MODEL_PATH}'")
        print("    Please run:  python main.py --train\n")
        sys.exit(1)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
    model.eval()

    real_mean = torch.tensor(np.load(f"{MODEL_PATH}/real_mean.npy"), dtype=torch.float32)
    real_std  = torch.tensor(np.load(f"{MODEL_PATH}/real_std.npy"),  dtype=torch.float32)

    print("      ✅ Model + spectral statistics loaded!")
    return feature_extractor, model, real_mean, real_std


def _record_voice():
    import sounddevice as sd
    import soundfile as sf

    print(f"\n[2/3] Recording your voice for {RECORD_SECONDS} seconds...")
    for i in range(3, 0, -1):
        print(f"      Starting in {i}...")
        time.sleep(1)
    print("      🔴 RECORDING — speak now!\n")

    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio_array = audio.flatten()
    sf.write("recorded_voice.wav", audio_array, SAMPLE_RATE)
    print("      ⏹️  Done recording.")
    return audio_array


def _analyse(audio_array, feature_extractor, model, real_mean, real_std):
    from transformers import AutoFeatureExtractor  # already imported but kept for clarity

    print("\n[3/3] Analysing using mathematical formulas...")

    inputs = feature_extractor(
        audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        padding=True, max_length=SAMPLE_RATE * 5, truncation=True,
    )
    input_values = inputs["input_values"]

    with torch.no_grad():
        outputs = model(input_values=input_values)
        logits  = outputs.logits
        probs   = F.softmax(logits, dim=-1)
        p_real  = probs[:, 0]
        p_fake  = probs[:, 1]

        aas = compute_AAS(p_real, p_fake)
        sdi = compute_SDI(input_values, real_mean)
        tcs = compute_TCS(input_values, real_std)
        fdi = compute_FDI(aas, sdi, tcs)

    return {
        "p_real": p_real.item(),
        "p_fake": p_fake.item(),
        "aas":    aas.item(),
        "sdi":    sdi.item(),
        "tcs":    tcs.item(),
        "fdi":    fdi.item(),
    }


def _display_result(scores):
    fdi = scores["fdi"]
    sir_code, sir_label = classify_SIR(fdi)

    print("\n" + "=" * 55)
    print("         DETECTION RESULT")
    print("=" * 55)
    print(f"\n  {sir_label}")
    print()
    print(f"  ── Mathematical Scores ──────────────────────")
    print(f"  AAS  (Authenticity Score) : {scores['aas']:+.4f}")
    print(f"       α·P_real − β·P_fake − γ·H(p)")
    print()
    print(f"  SDI  (Spectral Deviation) :  {scores['sdi']:.4f}")
    print(f"       mean|features − μ_real|")
    print()
    print(f"  TCS  (Temporal Consistency): {scores['tcs']:.4f}")
    print(f"       1 − σ_input / (σ_real + ε)")
    print()
    print(f"  FDI  (Fraud Detection Index): {fdi:.4f}")
    print(f"       w1·AAS + w2·(1−SDI) + w3·TCS")
    print()
    print(f"  ── Raw Probabilities ────────────────────────")
    print(f"  Real Voice     : {scores['p_real']*100:.1f}%")
    print(f"  Deepfake Voice : {scores['p_fake']*100:.1f}%")
    print()
    print(f"  ── SIR Stage ────────────────────────────────")
    print(f"  {sir_label}")
    if sir_code == "S":
        print("  ✅ Voice is GENUINE — safe to proceed.")
    elif sir_code == "I":
        print("  ⚠️  Voice is SUSPICIOUS — manual review recommended.")
    else:
        print("  🚨 DEEPFAKE DETECTED — block / flag this voice!")
    print("=" * 55 + "\n")


def detect():
    feature_extractor, model, real_mean, real_std = _load_model()

    print("\n" + "="*55)
    print("  DEEPFAKE AUDIO DETECTION — LIVE MIC")
    print("="*55)

    while True:
        print("\n─────────────────────────────────────")
        print("  Press ENTER to record & analyse")
        print("  Type  'q'    to quit")
        print("─────────────────────────────────────")
        cmd = input("  > ").strip().lower()
        if cmd == "q":
            print("\n  Goodbye!\n")
            break

        audio_array = _record_voice()
        scores      = _analyse(audio_array, feature_extractor, model, real_mean, real_std)
        _display_result(scores)


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Audio Detection — train or detect"
    )
    parser.add_argument("--train",  action="store_true",
                        help="Train the model from scratch")
    parser.add_argument("--detect", action="store_true",
                        help="Run live microphone detection")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.detect:
        detect()
    else:
        # Auto mode: train if no model exists, else detect
        if os.path.exists(MODEL_PATH):
            print(f"  ✅ Found existing model at '{MODEL_PATH}' — starting detection.")
            detect()
        else:
            print(f"  ℹ️  No model found at '{MODEL_PATH}' — starting training first.")
            train()
            detect()


if __name__ == "__main__":
    main()
