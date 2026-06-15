"""
predict.py
==========
Inference library for the Ticket Severity Mismatch Detector
("Hidden Crisis" / "False Alarm" auditor).

This module is the single source of truth for:
  - text preprocessing (must match training exactly),
  - the keyword rule-scorer,
  - the DistilBERT + LoRA + metadata model architecture,
  - loading the saved artifacts and predicting on one ticket.

train_pipeline.py imports the architecture / preprocessing from here so the
trained model.pt always loads back into an identical graph. app.py imports
the TicketPredictor class from here so the deployed UI and the CLI share one
inference path.

Metadata features (4): channel, category, resolution_time_norm, priority_norm.
The assigned priority is included because the mismatch label is defined relative
to it; it is supplied per ticket at inference, so there is no train/serve skew.

Saved artifacts expected under --model-dir (default: saved_model/):
    model.pt                     -> torch state_dict of the full model
    tokenizer/                   -> distilbert tokenizer files
    channel_encoder.pkl          -> sklearn LabelEncoder (Ticket_Channel)
    category_encoder.pkl         -> sklearn LabelEncoder (Issue_Category)
    resolution_scaler.pkl        -> sklearn MinMaxScaler (Resolution_Time_Hours)
    threshold.json   (optional)  -> validation-tuned decision threshold
    encoder_config/  (optional)  -> distilbert config for fully offline load

CLI:
    python predict.py --subject "Cannot login to account" \
        --description "Hi Support, our whole team is locked out and a customer breach is suspected." \
        --channel "Email" --category "Technical issue" \
        --resolution-hours 72 --priority "Low"
"""

import os
import re
import json
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn

# Keep noisy library warnings out of a clean CLI / app log.
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Constants (must stay identical to the training notebook)
# --------------------------------------------------------------------------- #
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
N_METADATA = 4  # CHANGED: 3 -> 4 (added normalized assigned priority)
N_CLASSES = 2

# LoRA hyper-parameters used when the model was trained.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_lin", "v_lin"]

DROPOUT = 0.3

REQUIRED_COLUMNS = [
    "Ticket_ID",
    "Ticket_Subject",
    "Ticket_Description",
    "Priority_Level",
    "Ticket_Channel",
    "Issue_Category",
    "Resolution_Time_Hours",
    "Customer_Email",
]

PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_INVERSE = {v: k for k, v in PRIORITY_MAP.items()}

# Binary head output meaning.
MODEL_LABELS = {0: "Consistent", 1: "Mismatch"}

# Final, human-facing verdict labels.
VERDICT_CONSISTENT = "Consistent"
VERDICT_HIDDEN_CRISIS = "Hidden Crisis"
VERDICT_FALSE_ALARM = "False Alarm"

# Keyword lexicons copied verbatim from the notebook's pseudo-label stage.
CRITICAL_WORDS = [
    "outage",
    "security",
    "fraud",
    "breach",
    "data loss",
    "stolen card",
    "unauthorized",
    "cannot login",
    "system down",
    "payment failed",
    "account hacked",
    "service unavailable",
    "data corruption",
]

HIGH_WORDS = [
    "crash",
    "error",
    "failed",
    "sync",
    "invoice discrepancy",
    "login issue",
    "payment issue",
    "screen freezes",
    "api error",
    "application crash",
    "data not syncing",
]


# --------------------------------------------------------------------------- #
# Preprocessing (reproduces clean_ticket_text + clean_text from the notebook)
# --------------------------------------------------------------------------- #
def strip_boilerplate(text):
    """Notebook clean_ticket_text: drop a leading 'Hi Support,' and keep the
    first sentence only."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    text = re.sub(r"^Hi Support,\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"[?.!]", text)[0]
    return text.strip()


def normalize_text(text):
    """Notebook clean_text: lowercase, strip urls, keep alphanumerics, collapse
    whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_clean_text(subject, description):
    """Reproduce the exact field the model was trained on:

        clean_description = strip_boilerplate(description)
        combined          = subject + " " + clean_description
        clean_text        = normalize_text(combined)
    """
    subject = "" if subject is None else str(subject)
    clean_description = strip_boilerplate(description)
    combined = (subject if subject else "") + " " + (clean_description if clean_description else "")
    return normalize_text(combined)


def rule_score(text):
    """Notebook rule_score: keyword-weighted severity in {0,1,2,3} plus the list
    of matched evidence terms. Runs on normalized clean_text."""
    score = 0
    evidence = []

    for word in CRITICAL_WORDS:
        if word in text:
            score += 3
            evidence.append(word)

    for word in HIGH_WORDS:
        if word in text:
            score += 2
            evidence.append(word)

    if score >= 6:
        severity = 3
    elif score >= 4:
        severity = 2
    elif score >= 2:
        severity = 1
    else:
        severity = 0

    return severity, evidence


# --------------------------------------------------------------------------- #
# Model architecture (identical to DistilBERTLoRAWithMetadata in the notebook)
# --------------------------------------------------------------------------- #
def build_lora_config():
    """Recreate the training LoRA config. Imported here so train_pipeline and
    inference build the same adapter key names."""
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
    )


class DistilBERTLoRAWithMetadata(nn.Module):
    """DistilBERT encoder (LoRA-adapted) fused with a small metadata MLP.

    forward inputs:
        input_ids, attention_mask        : tokenized clean_text
        channel, category, res_time,
        priority                          : float metadata, stacked to (batch, 4)
    """

    def __init__(
        self,
        model_name=MODEL_NAME,
        lora_config=None,
        n_metadata=N_METADATA,
        n_classes=N_CLASSES,
        dropout=DROPOUT,
        base_config=None,
    ):
        super().__init__()
        from transformers import AutoModel
        from peft import get_peft_model

        if lora_config is None:
            lora_config = build_lora_config()

        # When base_config is provided we build the encoder from config only
        # (no pretrained-weight download); model.pt supplies the real weights.
        if base_config is not None:
            base_model = AutoModel.from_config(base_config)
        else:
            base_model = AutoModel.from_pretrained(model_name)

        self.encoder = get_peft_model(base_model, lora_config)

        hidden_size = base_model.config.hidden_size

        self.meta_proj = nn.Sequential(
            nn.Linear(n_metadata, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    # CHANGED: forward now accepts the priority metadata feature
    def forward(self, input_ids, attention_mask, channel, category, res_time, priority):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # CHANGED: stack 4 metadata features (priority added)
        meta = torch.stack([channel, category, res_time, priority], dim=1)
        meta_out = self.meta_proj(meta)

        combined = torch.cat([cls_output, meta_out], dim=1)
        return self.classifier(combined)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def safe_encode(encoder, value, default=0):
    """LabelEncoder.transform raises on unseen labels. In a live app the user
    can type any channel / category, so unseen values fall back to default."""
    value = str(value)
    classes = list(getattr(encoder, "classes_", []))
    if value in classes:
        return int(encoder.transform([value])[0])
    return default


def classify_verdict(mismatch_prob, rule_sev, assigned_num, res_norm, threshold=0.5):
    """Combine the trained model's binary decision with a deterministic safety
    rule, then assign direction.

    - The model decides mismatch vs consistent (validated to >= 0.83 accuracy).
    - A deterministic guard treats any large keyword-vs-priority gap
      (|delta| >= 2) as a mismatch too, so blatant adversarial tickets never
      slip through on a borderline model score.
    - Direction follows the notebook sign convention:
          inferred severity > assigned -> under-prioritized -> Hidden Crisis
          inferred severity < assigned -> over-prioritized  -> False Alarm
    """
    delta = rule_sev - assigned_num
    model_flag = 1 if mismatch_prob >= threshold else 0
    final_mismatch = (model_flag == 1) or (abs(delta) >= 2)

    if not final_mismatch:
        verdict = VERDICT_CONSISTENT
    else:
        if delta > 0:
            verdict = VERDICT_HIDDEN_CRISIS
        elif delta < 0:
            verdict = VERDICT_FALSE_ALARM
        else:
            # Keyword severity ties the assigned priority: break the tie with
            # resolution time (long handling time leans under-prioritized).
            verdict = VERDICT_HIDDEN_CRISIS if res_norm >= 0.5 else VERDICT_FALSE_ALARM

    return verdict, delta, model_flag, final_mismatch


def build_dossier(result):
    """Plain-text, 3-point explanation mirroring the Mistral dossier structure
    from the notebook. No LLM required, so it ships inside the app."""
    verdict = result["verdict"]
    assigned = result["rule"]["assigned_priority"]
    inferred = result["rule"]["inferred_severity"]
    inferred_name = PRIORITY_INVERSE.get(inferred, str(inferred))
    evidence = result["rule"]["evidence"]
    res_hours = result["input"]["resolution_hours"]

    evidence_str = ", ".join(evidence) if evidence else "no critical keyword evidence"

    if verdict == VERDICT_CONSISTENT:
        return (
            f"The assigned priority ({assigned}) is consistent with the inferred "
            f"severity ({inferred_name}). Signals reviewed: keyword evidence "
            f"({evidence_str}) and resolution time ({res_hours} hours). No "
            f"priority correction is recommended."
        )

    if verdict == VERDICT_HIDDEN_CRISIS:
        reason = (
            f"The ticket was logged as {assigned} but the content points to a "
            f"higher severity ({inferred_name}), so it is likely under-prioritized."
        )
    else:  # False Alarm
        reason = (
            f"The ticket was logged as {assigned} but the content points to a "
            f"lower severity ({inferred_name}), so it is likely over-prioritized."
        )

    return (
        f"{reason} The inferred severity is supported by keyword evidence "
        f"({evidence_str}) and a resolution time of {res_hours} hours. It is "
        f"therefore flagged as a {verdict}."
    )


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class TicketPredictor:
    """Loads the saved artifacts once and scores tickets."""

    def __init__(self, model_dir="saved_model", device=None):
        self.model_dir = model_dir
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.default_threshold = 0.5  # CHANGED: may be overridden by threshold.json
        self._load()

    def _load(self):
        import joblib
        from transformers import AutoTokenizer, AutoConfig

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                f"Run train_pipeline.py first to create it."
            )

        tok_dir = os.path.join(self.model_dir, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)

        self.channel_encoder = joblib.load(os.path.join(self.model_dir, "channel_encoder.pkl"))
        self.category_encoder = joblib.load(os.path.join(self.model_dir, "category_encoder.pkl"))
        self.resolution_scaler = joblib.load(os.path.join(self.model_dir, "resolution_scaler.pkl"))

        # Prefer a locally-saved encoder config for a fully offline load; the
        # real weights come from model.pt regardless.
        cfg_dir = os.path.join(self.model_dir, "encoder_config")
        if os.path.isdir(cfg_dir):
            base_config = AutoConfig.from_pretrained(cfg_dir)
        else:
            base_config = AutoConfig.from_pretrained(MODEL_NAME)

        self.model = DistilBERTLoRAWithMetadata(
            model_name=MODEL_NAME,
            lora_config=build_lora_config(),
            base_config=base_config,
        )

        state_path = os.path.join(self.model_dir, "model.pt")
        state = torch.load(state_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)

        # The head and metadata projection must be present; the base encoder may
        # show benign buffer name differences across library versions.
        critical_missing = [
            k for k in missing if k.startswith("classifier.") or k.startswith("meta_proj.")
        ]
        if critical_missing:
            raise RuntimeError(
                "Saved model is incompatible with the current architecture. "
                f"Missing critical keys: {critical_missing[:6]} ..."
            )

        # CHANGED: load the validation-tuned decision threshold if present.
        # Absent file -> keep 0.5, so nothing breaks on an older artifact set.
        thr_path = os.path.join(self.model_dir, "threshold.json")
        if os.path.isfile(thr_path):
            try:
                with open(thr_path) as f:
                    self.default_threshold = float(json.load(f).get("threshold", 0.5))
            except Exception:
                self.default_threshold = 0.5

        self.model.to(self.device)
        self.model.eval()

    def predict(self, ticket, threshold=None):
        """Score one ticket.

        ticket: dict with keys
            subject, description, channel, category,
            resolution_hours, priority
        threshold: if None, uses the saved/tuned default_threshold.
        Returns a structured result dict.
        """
        subject = ticket.get("subject", "")
        description = ticket.get("description", "")
        channel = ticket.get("channel", "")
        category = ticket.get("category", "")
        resolution_hours = ticket.get("resolution_hours", 0.0)
        priority = ticket.get("priority", "Low")

        try:
            resolution_hours = float(resolution_hours)
        except (TypeError, ValueError):
            resolution_hours = 0.0

        # CHANGED: resolve the threshold (explicit arg wins, else tuned default)
        thr = self.default_threshold if threshold is None else threshold

        clean_text = build_clean_text(subject, description)

        # --- text branch ---
        encoding = self.tokenizer(
            clean_text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # --- metadata branch (replicates training feature construction) ---
        channel_code = float(safe_encode(self.channel_encoder, channel))
        category_code = float(safe_encode(self.category_encoder, category))
        res_norm = float(self.resolution_scaler.transform([[resolution_hours]])[0][0])
        # CHANGED: normalized assigned priority, matching training (priority_num / 3.0)
        priority_norm = PRIORITY_MAP.get(str(priority), 0) / 3.0

        channel_t = torch.tensor([channel_code], dtype=torch.float, device=self.device)
        category_t = torch.tensor([category_code], dtype=torch.float, device=self.device)
        res_t = torch.tensor([res_norm], dtype=torch.float, device=self.device)
        priority_t = torch.tensor([priority_norm], dtype=torch.float, device=self.device)  # CHANGED

        with torch.no_grad():
            # CHANGED: pass the priority feature to the model
            logits = self.model(input_ids, attention_mask, channel_t, category_t, res_t, priority_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        mismatch_prob = float(probs[1])
        consistent_prob = float(probs[0])

        # --- rule severity + verdict ---
        assigned_num = PRIORITY_MAP.get(str(priority), 0)
        rule_sev, evidence = rule_score(clean_text)

        verdict, delta, model_flag, final_mismatch = classify_verdict(
            mismatch_prob, rule_sev, assigned_num, res_norm, threshold=thr
        )

        model_conf = mismatch_prob if final_mismatch else consistent_prob
        rule_conf = round(min(0.99, 0.5 + abs(delta) * 0.15), 2)

        result = {
            "input": {
                "subject": subject,
                "description": description,
                "channel": channel,
                "category": category,
                "resolution_hours": resolution_hours,
                "priority": priority,
            },
            "clean_text": clean_text,
            "model": {
                "label": MODEL_LABELS[model_flag],
                "mismatch_probability": round(mismatch_prob, 4),
                "consistent_probability": round(consistent_prob, 4),
                "confidence": round(model_conf, 4),
            },
            "rule": {
                "inferred_severity": int(rule_sev),
                "evidence": evidence,
                "assigned_priority": str(priority),
                "assigned_priority_num": int(assigned_num),
                "severity_delta": int(delta),
                "resolution_time_norm": round(res_norm, 4),
                "rule_confidence": rule_conf,
            },
            "verdict": verdict,
            "is_mismatch": bool(final_mismatch),
        }
        result["dossier"] = build_dossier(result)
        return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(
        description="Score a support ticket for severity mismatch "
        "(Hidden Crisis / False Alarm)."
    )
    p.add_argument("--model-dir", default="saved_model", help="Path to saved artifacts.")
    p.add_argument("--ticket", help="Path to a JSON file describing one ticket.")
    p.add_argument("--subject", default="", help="Ticket subject.")
    p.add_argument("--description", default="", help="Ticket description.")
    p.add_argument("--channel", default="", help="Ticket channel.")
    p.add_argument("--category", default="", help="Issue category.")
    p.add_argument("--resolution-hours", type=float, default=0.0, help="Resolution time in hours.")
    p.add_argument(
        "--priority",
        default="Low",
        choices=list(PRIORITY_MAP.keys()),
        help="Human-assigned priority.",
    )
    # CHANGED: default None -> uses the tuned threshold saved in threshold.json
    p.add_argument("--threshold", type=float, default=None, help="Mismatch probability threshold.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.ticket:
        with open(args.ticket) as f:
            ticket = json.load(f)
    else:
        ticket = {
            "subject": args.subject,
            "description": args.description,
            "channel": args.channel,
            "category": args.category,
            "resolution_hours": args.resolution_hours,
            "priority": args.priority,
        }

    predictor = TicketPredictor(model_dir=args.model_dir)
    result = predictor.predict(ticket, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
