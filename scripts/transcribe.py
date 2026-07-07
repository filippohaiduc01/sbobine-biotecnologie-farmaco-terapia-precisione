"""
╔══════════════════════════════════════════════════════════════════╗
║     TRASCRIZIONE AUDIO → PDF  |  Versione 3.0 (Due Fasi)         							║
╠══════════════════════════════════════════════════════════════════╣
║   Scarica Python 3.11.9 (Windows 64-bit)                       						    ║
║                                                                  							║
║    INSTALLAZIONE (terminale PyCharm con (.venv) attivo):         							║
║    pip install whisperx reportlab demucs torch torchaudio        							║
║    pip install soundfile                                         							║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
import subprocess
import numpy as np
import whisperx
import torch       
import gc          
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE
# ════════════════════════════════════════════════════════════════

FILES = [
    "[nome.audio]",
]

MODEL_NAME = "large-v3"

PROMPT = (
    "[nome del corso]"
    "[concetti chiave per i vari corsi]"
)

BEAM_SIZE   = 10
BEST_OF     = 10
TEMPERATURE = 0.0
BATCH_SIZE  = 8

RIDUCI_RUMORE_DEMUCS = True
ALLINEAMENTO = True

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

OUTPUT_PDF = "[nome del file pdf da creare]"

# ════════════════════════════════════════════════════════════════

def separa_voce_demucs(percorso: str) -> str:
    """Usa Demucs per separare la voce dal rumore di fondo."""
    cartella_out = "demucs_output"

    comando = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "--out", cartella_out,
        "--name", "htdemucs_ft",
        percorso
    ]

    risultato = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if risultato.returncode != 0:
        print(f"    ⚠ Demucs fallito, uso il file originale.")
        # Mostriamo gli ultimi 800 caratteri dell'errore (la parte utile, ignorando la barra di caricamento)
        messaggio_errore = risultato.stderr[-800:] if risultato.stderr else "Nessun output"
        print(f"      Errore:\n{messaggio_errore}")
        return percorso

    nome_base = os.path.splitext(os.path.basename(percorso))[0]
    vocals_path = os.path.join(cartella_out, "htdemucs_ft", nome_base, "vocals.wav")

    if not os.path.exists(vocals_path):
        print(f"    ⚠ File vocals non trovato, uso il file originale.")
        return percorso

    print(f"    ✔ Voce isolata con successo.")
    return vocals_path


def trascrivi_file(model, percorso: str) -> dict:
    """Trascrive un file audio con WhisperX."""
    print(f"    → Caricamento audio in memoria ...")
    audio = whisperx.load_audio(percorso)

    print(f"    → Trascrizione in corso ...")
    result = model.transcribe(
        audio,
        batch_size=BATCH_SIZE,
        language="it"
    )

    if ALLINEAMENTO:
        print(f"    → Allineamento parola per parola ...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code="it",
                device=DEVICE
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
        except Exception as e:
            print(f"    ⚠ Allineamento fallito (non critico): {e}")

    return result


def estrai_testo(result: dict) -> str:
    if "segments" in result:
        return " ".join(seg["text"].strip() for seg in result["segments"] if seg.get("text", "").strip())
    return result.get("text", "")


def build_pdf(trascrizioni: dict, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    heading_style = ParagraphStyle("LessonHeading", parent=styles["Heading1"], fontSize=15, spaceBefore=18, spaceAfter=6, textColor=colors.HexColor("#16213e"))
    body_style = ParagraphStyle("LessonBody", parent=styles["Normal"], fontSize=11, leading=17, spaceAfter=8, textColor=colors.HexColor("#333333"))
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#888888"), spaceAfter=4)

    story = [
        Spacer(1, 3 * cm),
        Paragraph("Trascrizioni Lezioni", title_style),
        Paragraph("[nome del prof/prof.ssa]", heading_style),
        Spacer(1, 0.5 * cm),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Modello: WhisperX {MODEL_NAME} | Demucs: {'Sì' if RIDUCI_RUMORE_DEMUCS else 'No'} | File: {len(trascrizioni)}", meta_style),
        PageBreak()
    ]

    for i, (nome_file, testo) in enumerate(trascrizioni.items()):
        nome_display = os.path.splitext(os.path.basename(nome_file))[0].replace("_", " ").replace("  ", " – ")
        story.extend([Paragraph(nome_display, heading_style), HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")), Spacer(1, 0.3 * cm)])

        frasi = [f.strip() for f in testo.replace("\n", " ").split(".") if f.strip()]
        chunk = []
        for j, frase in enumerate(frasi):
            chunk.append(frase + ".")
            if len(chunk) >= 5 or j == len(frasi) - 1:
                story.append(Paragraph(" ".join(chunk), body_style))
                chunk = []
        if i < len(trascrizioni) - 1:
            story.append(PageBreak())

    doc.build(story)


def main():
    print("=" * 60)
    print("  Trascrizione WhisperX | Versione 3.0 (Due Fasi)")
    print("=" * 60)

    trovati = [f for f in FILES if os.path.exists(f)]
    for f in FILES:
        if f in trovati:
            print(f"  ✔ Trovato: {f}")
        else:
            print(f"  ✘ Non trovato (skip): {f}")

    if not trovati:
        print("\n❌ Nessun file trovato.")
        return

    # ==========================================
    # FASE 1: PULIZIA AUDIO (DEMUCS)
    # ==========================================
    file_da_trascrivere = []

    if RIDUCI_RUMORE_DEMUCS:
        print("\n" + "=" * 60)
        print("  FASE 1/2: Separazione Voce (Demucs)")
        print("=" * 60)
        for idx, f in enumerate(trovati, 1):
            print(f"\n[{idx}/{len(trovati)}] Elaborazione Demucs per: {f}")
            file_pulito = separa_voce_demucs(f)
            file_da_trascrivere.append((f, file_pulito))
    else:
        file_da_trascrivere = [(f, f) for f in trovati]

# ==========================================
# FASE 2: TRASCRIZIONE (WHISPERX)
# ==========================================
    print("\n" + "=" * 60)
    print("  FASE 2/2: Caricamento IA e Trascrizione (WhisperX)")
    print("=" * 60)

    print(f"Caricamento modello '{MODEL_NAME}' su {DEVICE}...")
    model = whisperx.load_model(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, language="it")
    print("✔ Modello caricato con successo.\n")

    trascrizioni = {}
    for idx, (nome_originale, percorso_pulito) in enumerate(file_da_trascrivere, 1):
        print(f"[{idx}/{len(file_da_trascrivere)}] Trascrivendo: {nome_originale}")

        result = trascrivi_file(model, percorso_pulito)
        testo = estrai_testo(result)
        trascrizioni[nome_originale] = testo
        print(f"    ✔ {len(testo)} caratteri trascritti.\n")

        # Salva TXT di backup
        nome_txt = os.path.splitext(nome_originale)[0] + "_whisperx.txt"
        with open(nome_txt, "w", encoding="utf-8") as f:
            f.write(testo)

        # ====================================================
        # AGGIUNTA: PULIZIA MEMORIA VRAM PER EVITARE CRASH
        # ====================================================
        del result
        gc.collect()
        torch.cuda.empty_cache()
        # ====================================================

    print("Generazione PDF in corso...")
    build_pdf(trascrizioni, OUTPUT_PDF)
    print(f"\n✅ PDF salvato in: {os.path.abspath(OUTPUT_PDF)}")
    print("=" * 60)


if __name__ == "__main__":
    main()