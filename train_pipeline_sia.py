"""
train_pipeline.py
=================
End-to-end training pipeline for the Ticket Severity Mismatch Detector.

It reproduces the notebook exactly, in ordered stages, and writes the artifacts
that predict.py / app.py load at inference time:

    saved_model/model.pt
    saved_model/tokenizer/
    saved_model/channel_encoder.pkl
    saved_model/category_encoder.pkl
    saved_model/resolution_scaler.pkl
    saved_model/encoder_config/        (for fully offline inference)
    saved_model/threshold.json         (validation-tuned decision threshold)
    saved_model/results.csv
    saved_model/training_history.csv

Stages (each caches to disk and is skipped on re-run unless --force):
    1. Preprocess + feature engineering        -> feature_engineered.csv
    2. LLM severity scoring (Mistral, optional) -> mistral_scores.csv
    3. Fusion + pseudo-label / mismatch typing  -> pseudo_labeled_dataset.csv
    4. Split + encode + scale                   -> train.csv / val.csv / test.csv
    5. Train DistilBERT + LoRA + metadata head  -> saved_model/
    6. Dossier generation (Mistral, optional)   -> sample_dossiers.json

The Mistral 7B stages are heavy and gated behind flags. Without --use-llm the
pipeline still runs end to end on CPU/GPU by substituting a rule-based proxy for
the LLM severity signal (a clear warning is printed). The trained DistilBERT
model and the deployable app do NOT need Mistral.

Metadata features (4): channel, category, resolution_time_norm, priority_norm.
The assigned priority is a feature because the mismatch label is defined relative
to it; it is available at inference, so there is no train/serve skew.

Usage:
    python train_pipeline.py --data customer_support_tickets.csv
    python train_pipeline.py --data customer_support_tickets.csv --use-llm
    python train_pipeline.py --data customer_support_tickets.csv --force --epochs 8
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from predict import (
    REQUIRED_COLUMNS,
    PRIORITY_MAP,
    MODEL_NAME,
    MAX_LEN,
    strip_boilerplate,
    normalize_text,
    rule_score,
    build_lora_config,
    DistilBERTLoRAWithMetadata,
)

warnings.filterwarnings("ignore")

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


# --------------------------------------------------------------------------- #
# Stage 1: Preprocess + feature engineering
# --------------------------------------------------------------------------- #
def stage_feature_engineering(data_path, out_path, force=False):
    if os.path.exists(out_path) and not force:
        print(f"[stage 1] Found {out_path}, skipping (use --force to rebuild).")
        return pd.read_csv(out_path)

    print(f"[stage 1] Loading {data_path}")
    df = pd.read_csv(data_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    print(f"[stage 1] Shape after column select: {df.shape}")

    # Numeric safety for resolution time.
    df["Resolution_Time_Hours"] = pd.to_numeric(
        df["Resolution_Time_Hours"], errors="coerce"
    )
    median_res = df["Resolution_Time_Hours"].median()
    df["Resolution_Time_Hours"] = df["Resolution_Time_Hours"].fillna(median_res)

    # Text cleaning (notebook clean_ticket_text + clean_text).
    df["clean_description"] = df["Ticket_Description"].apply(strip_boilerplate)
    df["clean_text"] = df["Ticket_Subject"].fillna("") + " " + df["clean_description"].fillna("")
    df["clean_text"] = df["clean_text"].apply(normalize_text)

    # Rule keyword severity.
    rule_out = df["clean_text"].apply(lambda x: pd.Series(_rule_score_row(x)))
    df["rule_score"] = rule_out[0]
    df["rule_evidence"] = rule_out[1]

    # Resolution-time -> priority regressor (RandomForest), as in the notebook.
    from sklearn.ensemble import RandomForestRegressor

    df["priority_num"] = df["Priority_Level"].map(PRIORITY_MAP)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(df[["Resolution_Time_Hours"]], df["priority_num"])
    pred = rf.predict(df[["Resolution_Time_Hours"]])
    df["resolution_score"] = pred.clip(0, 3).round().astype(int)

    # Sentence embeddings + KMeans cluster scoring.
    print("[stage 1] Encoding embeddings (all-MiniLM-L6-v2)")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(
        df["clean_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("[stage 1] Selecting best k for KMeans")
    silhouette_scores = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(
            embeddings, labels, sample_size=min(2000, len(df)), random_state=42
        )
        silhouette_scores[k] = score
        print(f"          k={k} -> {score:.4f}")
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"[stage 1] Best k = {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)

    cluster_avg_resolution = (
        df.groupby("cluster")["Resolution_Time_Hours"].mean().sort_values()
    )
    cluster_score_map = {
        cluster: rank for rank, cluster in enumerate(cluster_avg_resolution.index)
    }
    df["cluster_score"] = df["cluster"].map(cluster_score_map)

    df.to_csv(out_path, index=False)
    print(f"[stage 1] Saved {out_path}")
    return df


def _rule_score_row(text):
    severity, evidence = rule_score(text)
    return severity, ",".join(evidence)


# --------------------------------------------------------------------------- #
# Stage 2: LLM severity scoring (optional, Mistral)
# --------------------------------------------------------------------------- #
def build_llm_prompt(subject, description, category, channel):
    return f"""
You are a support ticket severity auditor.

Assign a severity score.

Severity scale:

0 = Low
1 = Medium
2 = High
3 = Critical

Definitions:

3:
Security incidents,
fraud,
account compromise,
service outage,
major operational disruption.

2:
Application crashes,
API failures,
login failures,
payment failures,
data synchronization failures.

1:
User-impacting issues with workarounds.

0:
Questions,
feature requests,
account updates,
minor requests.

Ticket Category:
{category}

Ticket Channel:
{channel}

Subject:
{subject}

Description:
{description}

Return ONLY ONE NUMBER.

Severity Score:
"""


def stage_llm_scores(df, out_path, use_llm=False, force=False, save_every=500):
    """Produce mistral_scores.csv (Ticket_ID, llm_score).

    Priority:
      1. cached file (always reused unless --force),
      2. real Mistral inference if --use-llm,
      3. rule-based proxy otherwise (warned).
    """
    if os.path.exists(out_path) and not force:
        print(f"[stage 2] Found {out_path}, reusing cached LLM scores.")
        return pd.read_csv(out_path)

    if not use_llm:
        print(
            "[stage 2] WARNING: running without Mistral (--use-llm not set). "
            "Substituting rule-based severity as the llm_score proxy. Fusion "
            "results will differ from the LLM-labelled run."
        )
        proxy = df[["Ticket_ID", "rule_score"]].rename(columns={"rule_score": "llm_score"})
        proxy.to_csv(out_path, index=False)
        return proxy

    import re
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[stage 2] Loading {LLM_MODEL_NAME} (this needs a GPU and is slow)")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    def predict_severity(subject, description, category, channel):
        prompt = build_llm_prompt(subject, description, category, channel)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        numbers = re.findall(r"\b([0-3])\b", response)
        return int(numbers[0]) if numbers else None

    checkpoint_file = "mistral_scores_checkpoint.csv"
    if os.path.exists(checkpoint_file):
        checkpoint = pd.read_csv(checkpoint_file).drop_duplicates(
            subset="Ticket_ID", keep="last"
        )
        completed_ids = set(checkpoint["Ticket_ID"])
        remaining_df = df[~df["Ticket_ID"].isin(completed_ids)].copy()
        results = checkpoint.to_dict("records")
        print(f"[stage 2] Resuming: {len(checkpoint)} done, {len(remaining_df)} remaining")
    else:
        remaining_df = df.copy()
        results = []
        print(f"[stage 2] Starting fresh on {len(df)} tickets")

    counter = 0
    for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df)):
        try:
            score = predict_severity(
                row["Ticket_Subject"],
                row["clean_description"],
                row["Issue_Category"],
                row["Ticket_Channel"],
            )
        except Exception:
            score = None
        results.append({"Ticket_ID": row["Ticket_ID"], "llm_score": score})
        counter += 1
        if counter % save_every == 0:
            pd.DataFrame(results).drop_duplicates(
                subset="Ticket_ID", keep="last"
            ).to_csv(checkpoint_file, index=False)
            print(f"[stage 2] Checkpoint at {counter}")

    final_df = pd.DataFrame(results).drop_duplicates(subset="Ticket_ID", keep="last")
    final_df.to_csv(out_path, index=False)
    print(f"[stage 2] Saved {out_path}")
    return final_df


# --------------------------------------------------------------------------- #
# Stage 3: Fusion + pseudo-labels + mismatch typing
# --------------------------------------------------------------------------- #
def stage_fusion(features, llm, out_path, force=False):
    if os.path.exists(out_path) and not force:
        print(f"[stage 3] Found {out_path}, skipping.")
        return pd.read_csv(out_path)

    df = features.merge(llm, on="Ticket_ID", how="left")

    # If any LLM score is missing, fall back to the rule score for that row so
    # the fusion stays defined.
    df["llm_score"] = df["llm_score"].fillna(df["rule_score"])

    df["severity_fusion"] = (
        0.40 * df["llm_score"]
        + 0.30 * df["cluster_score"]
        + 0.20 * df["resolution_score"]
        + 0.10 * df["rule_score"]
    )

    df["inferred_severity"] = 0
    df.loc[df["severity_fusion"] >= 0.75, "inferred_severity"] = 1
    df.loc[df["severity_fusion"] >= 1.50, "inferred_severity"] = 2
    df.loc[df["severity_fusion"] >= 2.25, "inferred_severity"] = 3

    df["assigned_priority_num"] = df["Priority_Level"].map(PRIORITY_MAP)
    # CHANGED: normalized assigned priority feature (Low=0.00 .. Critical=1.00).
    # Carried into the splits so the classifier can see the priority the mismatch
    # label is defined against.
    df["priority_norm"] = df["assigned_priority_num"] / 3.0
    df["severity_delta"] = df["inferred_severity"] - df["assigned_priority_num"]
    df["mismatch"] = (df["severity_delta"].abs() >= 2).astype(int)

    df["mismatch_type"] = "Consistent"
    df.loc[df["severity_delta"] >= 2, "mismatch_type"] = "Hidden Crisis"
    df.loc[df["severity_delta"] <= -2, "mismatch_type"] = "False Alarm"

    print("[stage 3] mismatch distribution:")
    print(df["mismatch"].value_counts())
    print("[stage 3] mismatch_type distribution:")
    print(df["mismatch_type"].value_counts())

    df.to_csv(out_path, index=False)
    print(f"[stage 3] Saved {out_path}")
    return df


# --------------------------------------------------------------------------- #
# Stage 4: Split, encode, scale
# --------------------------------------------------------------------------- #
def stage_prepare_dataset(df, out_dir, force=False):
    train_path = "train.csv"
    val_path = "val.csv"
    test_path = "test.csv"

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    import joblib

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]) and not force:
        print("[stage 4] Found train/val/test, skipping split.")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        # Encoders / scaler must still exist for inference.
        if not all(
            os.path.exists(os.path.join(out_dir, f))
            for f in ["channel_encoder.pkl", "category_encoder.pkl", "resolution_scaler.pkl"]
        ):
            print("[stage 4] Encoders missing; rebuilding from existing splits.")
            _fit_and_save_encoders(train_df, val_df, test_df, out_dir)
        return train_df, val_df, test_df

    # Guard stratification when a class is degenerate.
    stratify = df["mismatch"] if df["mismatch"].nunique() > 1 else None
    if stratify is None:
        print("[stage 4] WARNING: single mismatch class; disabling stratify.")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=stratify, random_state=42
    )
    stratify_temp = temp_df["mismatch"] if temp_df["mismatch"].nunique() > 1 else None
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=stratify_temp, random_state=42
    )

    train_df, val_df, test_df = _fit_and_save_encoders(train_df, val_df, test_df, out_dir)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[stage 4] Saved splits. Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")
    return train_df, val_df, test_df


def _fit_and_save_encoders(train_df, val_df, test_df, out_dir):
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    import joblib

    os.makedirs(out_dir, exist_ok=True)

    scaler = MinMaxScaler()
    train_df["resolution_time_norm"] = scaler.fit_transform(train_df[["Resolution_Time_Hours"]])
    val_df["resolution_time_norm"] = scaler.transform(val_df[["Resolution_Time_Hours"]])
    test_df["resolution_time_norm"] = scaler.transform(test_df[["Resolution_Time_Hours"]])

    # CHANGED: ensure the normalized priority feature exists on each split
    # (derive it if an older cached split lacks the column).
    for part in (train_df, val_df, test_df):
        if "priority_norm" not in part.columns:
            part["priority_norm"] = part["assigned_priority_num"].astype(float) / 3.0

    le_channel = LabelEncoder()
    train_df["channel_encoded"] = le_channel.fit_transform(train_df["Ticket_Channel"].astype(str))
    val_df["channel_encoded"] = _safe_transform(le_channel, val_df["Ticket_Channel"])
    test_df["channel_encoded"] = _safe_transform(le_channel, test_df["Ticket_Channel"])

    le_category = LabelEncoder()
    train_df["category_encoded"] = le_category.fit_transform(train_df["Issue_Category"].astype(str))
    val_df["category_encoded"] = _safe_transform(le_category, val_df["Issue_Category"])
    test_df["category_encoded"] = _safe_transform(le_category, test_df["Issue_Category"])

    joblib.dump(le_channel, os.path.join(out_dir, "channel_encoder.pkl"))
    joblib.dump(le_category, os.path.join(out_dir, "category_encoder.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "resolution_scaler.pkl"))
    print(f"[stage 4] Saved encoders + scaler to {out_dir}")
    return train_df, val_df, test_df


def _safe_transform(encoder, series):
    """Map unseen validation/test categories to 0 instead of raising."""
    classes = set(encoder.classes_)
    out = []
    for v in series.astype(str):
        out.append(int(encoder.transform([v])[0]) if v in classes else 0)
    return out


# --------------------------------------------------------------------------- #
# Stage 5: Train DistilBERT + LoRA + metadata head
# --------------------------------------------------------------------------- #
class TicketDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["clean_text"].astype(str).tolist()
        self.labels = df["mismatch"].tolist()
        self.channels = df["channel_encoded"].tolist()
        self.categories = df["category_encoded"].tolist()
        self.res_times = df["resolution_time_norm"].tolist()
        # CHANGED: normalized assigned priority feature (fallback derives it if the
        # column is absent, e.g. from an older cached split CSV).
        if "priority_norm" in df.columns:
            self.priorities = df["priority_norm"].tolist()
        else:
            self.priorities = (df["assigned_priority_num"].astype(float) / 3.0).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "channel": torch.tensor(self.channels[idx], dtype=torch.float),
            "category": torch.tensor(self.categories[idx], dtype=torch.float),
            "res_time": torch.tensor(self.res_times[idx], dtype=torch.float),
            # CHANGED: normalized assigned priority (4th metadata feature)
            "priority": torch.tensor(self.priorities[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    from sklearn.metrics import accuracy_score, f1_score

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []  # CHANGED: positive-class probabilities for threshold tuning

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            channel = batch["channel"].to(device)
            category = batch["category"].to(device)
            res_time = batch["res_time"].to(device)
            priority = batch["priority"].to(device)  # CHANGED: pull priority feature
            labels = batch["label"].to(device)

            if is_train:
                optimizer.zero_grad()

            # CHANGED: pass priority to the model
            logits = model(input_ids, attention_mask, channel, category, res_time, priority)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())  # CHANGED
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro"),
        all_preds,
        all_labels,
        all_probs,  # CHANGED: extra return value
    )


def _tune_threshold(val_probs, val_labels):
    """CHANGED: pick the validation threshold that clears all four verification
    gates (accuracy >= 0.83, macro-F1 >= 0.82, both recalls >= 0.78) with the
    HIGHEST accuracy; fall back to the macro-F1-optimal point if none clear them."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    val_probs = np.asarray(val_probs)
    val_labels = np.asarray(val_labels)

    acc_gate, f1_gate, rec_gate = 0.83, 0.82, 0.78
    best_threshold = 0.5
    best_passing_acc = -1.0
    best_macro = -1.0
    best_macro_threshold = 0.5

    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        preds_t = (val_probs >= t).astype(int)
        acc_t = accuracy_score(val_labels, preds_t)
        f1_t = f1_score(val_labels, preds_t, average="macro")
        rec_t = recall_score(val_labels, preds_t, average=None, zero_division=0)
        r0, r1 = (rec_t[0], rec_t[1]) if len(rec_t) == 2 else (0.0, 0.0)

        if f1_t > best_macro:
            best_macro = f1_t
            best_macro_threshold = float(t)

        if acc_t >= acc_gate and f1_t >= f1_gate and r0 >= rec_gate and r1 >= rec_gate:
            if acc_t > best_passing_acc:
                best_passing_acc = acc_t
                best_threshold = float(t)

    if best_passing_acc < 0:
        print(
            "[stage 5] No validation threshold cleared all four gates; "
            f"using macro-F1-optimal threshold {best_macro_threshold:.2f}."
        )
        return best_macro_threshold
    print(
        f"[stage 5] Tuned threshold (all gates pass, max accuracy): "
        f"{best_threshold:.2f} (val accuracy {best_passing_acc:.4f})"
    )
    return best_threshold


def stage_train(train_df, val_df, test_df, out_dir, epochs=8, batch_size=32, lr=2e-4):
    import copy  # CHANGED: snapshot best-validation weights
    from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, recall_score, accuracy_score, f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage 5] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = TicketDataset(train_df, tokenizer, MAX_LEN)
    val_ds = TicketDataset(val_df, tokenizer, MAX_LEN)
    test_ds = TicketDataset(test_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["mismatch"]),
        y=train_df["mismatch"],
    )
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"[stage 5] Class weights: {class_weights.tolist()}")

    base_config = AutoConfig.from_pretrained(MODEL_NAME)
    model = DistilBERTLoRAWithMetadata(
        model_name=MODEL_NAME,
        lora_config=build_lora_config(),
        base_config=base_config,
    ).to(device)

    # CHANGED: gentle label smoothing on top of the balanced class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # CHANGED: keep the best-by-validation-macro-F1 checkpoint
    best_val_f1 = -1.0
    best_state = None

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}\nEPOCH {epoch}/{epochs}\n{'=' * 50}")
        train_loss, train_acc, train_f1, _, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler
        )
        val_loss, val_acc, val_f1, _, _, _ = _run_epoch(model, val_loader, criterion, device)

        # CHANGED: snapshot weights whenever validation macro-F1 improves
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            print(f"  new best val Macro F1: {best_val_f1:.4f} (checkpoint kept)")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )
        print(f"Train -> Loss {train_loss:.4f} | Acc {train_acc:.4f} | Macro F1 {train_f1:.4f}")
        print(f"Val   -> Loss {val_loss:.4f} | Acc {val_acc:.4f} | Macro F1 {val_f1:.4f}")

    # CHANGED: restore the best-by-validation weights before tuning / test / saving
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[stage 5] Restored best checkpoint (val Macro F1 = {best_val_f1:.4f})")

    # CHANGED: tune the decision threshold on validation, then apply to test
    _, _, _, _, val_labels, val_probs = _run_epoch(model, val_loader, criterion, device)
    best_threshold = _tune_threshold(val_probs, val_labels)

    print("\n=== FINAL TEST EVALUATION ===")
    test_loss, _, _, _, test_labels, test_probs = _run_epoch(model, test_loader, criterion, device)
    test_labels = np.asarray(test_labels)
    test_probs = np.asarray(test_probs)
    test_preds = (test_probs >= best_threshold).astype(int)  # CHANGED: thresholded preds

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=["Consistent (0)", "Mismatch (1)"],
            digits=4,
        )
    )

    recalls = recall_score(test_labels, test_preds, average=None)
    print("\n=== VERIFICATION THRESHOLD CHECK ===")
    print(f"Accuracy >= 0.83: {'PASS' if test_acc >= 0.83 else 'FAIL'} ({test_acc:.4f})")
    print(f"Macro F1 >= 0.82: {'PASS' if test_f1 >= 0.82 else 'FAIL'} ({test_f1:.4f})")
    print(f"Recall class 0 >= 0.78: {'PASS' if recalls[0] >= 0.78 else 'FAIL'} ({recalls[0]:.4f})")
    print(f"Recall class 1 >= 0.78: {'PASS' if recalls[1] >= 0.78 else 'FAIL'} ({recalls[1]:.4f})")

    results_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Macro_F1", "Recall_Class0", "Recall_Class1"],
            "Value": [test_acc, test_f1, recalls[0], recalls[1]],
        }
    )
    history_df = pd.DataFrame(history)

    # CHANGED: pass the tuned threshold so it is persisted for inference
    _save_artifacts(model, tokenizer, base_config, results_df, history_df, out_dir, best_threshold)
    return model, results_df, history_df


def _save_artifacts(model, tokenizer, base_config, results_df, history_df, out_dir, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))

    # Save the encoder config so inference can build the base model fully
    # offline (no pretrained-weight download).
    base_config.save_pretrained(os.path.join(out_dir, "encoder_config"))

    results_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    history_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    # CHANGED: persist the validation-tuned decision threshold for predict.py
    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(threshold)}, f)

    print("\nTraining Complete")
    print(f"Model + artifacts saved to {out_dir}/")
    print("  - model.pt")
    print("  - tokenizer/")
    print("  - encoder_config/")
    print("  - channel_encoder.pkl, category_encoder.pkl, resolution_scaler.pkl")
    print("  - threshold.json")
    print("  - results.csv, training_history.csv")


# --------------------------------------------------------------------------- #
# Stage 6: Dossier generation (optional, Mistral)
# --------------------------------------------------------------------------- #
def stage_dossiers(df, out_path, use_llm=False, force=False, per_type=25):
    if os.path.exists(out_path) and not force:
        print(f"[stage 6] Found {out_path}, skipping.")
        return
    if not use_llm:
        print("[stage 6] Skipping LLM dossier generation (--use-llm not set).")
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    flagged = df[df["mismatch"] == 1].copy()
    hidden = flagged[flagged["mismatch_type"] == "Hidden Crisis"]
    false_alarm = flagged[flagged["mismatch_type"] == "False Alarm"]
    hidden = hidden.sample(min(per_type, len(hidden)), random_state=42)
    false_alarm = false_alarm.sample(min(per_type, len(false_alarm)), random_state=42)
    sample = pd.concat([hidden, false_alarm]).reset_index(drop=True)
    print(f"[stage 6] Generating dossiers for {len(sample)} tickets")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def build_prompt(row):
        return f"""
You are a support operations analyst.

Ticket Subject:
{row['Ticket_Subject']}

Ticket Description:
{row['Ticket_Description']}

Assigned Priority:
{row['Priority_Level']}

Inferred Severity:
{row['inferred_severity']}

Mismatch Type:
{row['mismatch_type']}

Keyword Evidence:
{row['rule_evidence']}

Resolution Time:
{row['Resolution_Time_Hours']} hours

Explain:

1. Why the assigned priority may not reflect the ticket urgency.
2. Which evidence supports the inferred severity.
3. Why the ticket is classified as {row['mismatch_type']}.

Use only the provided information.
Do not mention scores.
Do not invent information.
Maximum 3 sentences.
"""

    results = []
    for _, row in sample.iterrows():
        try:
            prompt = build_prompt(row)
            response = generator(prompt, max_new_tokens=120, do_sample=False, temperature=0.0)
            analysis = response[0]["generated_text"].replace(prompt, "").strip()
        except Exception:
            analysis = "Explanation generation failed."

        evidence = row["rule_evidence"]
        evidence = evidence if pd.notna(evidence) and str(evidence).strip() else "No keyword evidence"
        confidence = min(0.99, round(0.5 + abs(row["severity_delta"]) * 0.15, 2))

        results.append(
            {
                "ticket_id": row["Ticket_ID"],
                "assigned_priority": row["Priority_Level"],
                "inferred_severity": int(row["inferred_severity"]),
                "mismatch_type": row["mismatch_type"],
                "severity_delta": int(row["severity_delta"]),
                "constraint_analysis": analysis,
                "confidence": confidence,
            }
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[stage 6] Saved {out_path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description="Train the ticket severity mismatch detector.")
    p.add_argument("--data", default="customer_support_tickets.csv", help="Input CSV path.")
    p.add_argument("--out-dir", default="saved_model", help="Where to save artifacts.")
    p.add_argument("--epochs", type=int, default=8)  # CHANGED: 5 -> 8 (matches notebook)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--use-llm", action="store_true", help="Use Mistral 7B for severity scoring.")
    p.add_argument("--gen-dossiers", action="store_true", help="Generate LLM dossiers (needs --use-llm).")
    p.add_argument("--force", action="store_true", help="Recompute all cached stages.")
    return p.parse_args()


def main():
    args = _parse_args()

    features = stage_feature_engineering(
        args.data, "feature_engineered.csv", force=args.force
    )
    llm = stage_llm_scores(
        features, "mistral_scores.csv", use_llm=args.use_llm, force=args.force
    )
    fused = stage_fusion(features, llm, "pseudo_labeled_dataset.csv", force=args.force)
    train_df, val_df, test_df = stage_prepare_dataset(fused, args.out_dir, force=args.force)
    stage_train(
        train_df,
        val_df,
        test_df,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    if args.gen_dossiers:
        stage_dossiers(fused, "sample_dossiers.json", use_llm=args.use_llm, force=args.force)

    print("\nPipeline finished. You can now run:  python predict.py ...   or   streamlit run app.py")


if __name__ == "__main__":
    main()
