# Trascrizione Audio → PDF (WhisperX + Demucs) — Guida all'uso

Questo script prende uno o più file audio (es. registrazioni di lezioni), isola la voce dal rumore di fondo con **Demucs**, la trascrive con **WhisperX** e genera un **PDF** con tutte le trascrizioni.

---

## 1. Requisiti hardware — GPU

Questo script è pensato per girare su **GPU NVIDIA (CUDA)**, non su CPU. Sia Demucs che WhisperX in modalità `large-v3` sono pesanti:

- **GPU NVIDIA obbligatoria**, con almeno **6-8 GB di VRAM** per il modello `large-v3` (consigliati 8+ GB per stare tranquilli, soprattutto se l'audio è lungo).
- Serve un **driver NVIDIA aggiornato** e **CUDA** installato/compatibile con la versione di PyTorch che installerai.
- Se non hai una GPU NVIDIA (es. hai solo una CPU, o una GPU AMD/Intel, o un Mac), lo script **non funzionerà così com'è**: andrebbe modificato impostando `DEVICE = "cpu"` e `COMPUTE_TYPE = "int8"` (o `"float32"`), ma la trascrizione diventerà **molto più lenta** (anche 5-10 volte più lenta o peggio).
- Se hai poca VRAM, puoi usare un modello più piccolo (es. `"medium"` o `"small"` invece di `"large-v3"`), a scapito della qualità.

---

## 2. Installazione — Python

1. Scarica e installa **Python 3.11.9 (64-bit)** da [python.org](https://www.python.org/downloads/release/python-3119/) — su Windows spunta "Add python.exe to PATH" durante l'installazione.
2. Apri **PyCharm**, crea un nuovo progetto scegliendo come interprete Python 3.11.9.
3. Assicurati che il **venv** (ambiente virtuale) sia attivo — in basso a sinistra in PyCharm dovresti vedere `(.venv)` nel terminale integrato.

---

## 3. Installazione — Pacchetti Python

Apri il terminale di PyCharm (con `.venv` attivo) e installa i pacchetti:

```bash
pip install whisperx reportlab demucs torch torchaudio
pip install soundfile
```

### Nota importante su PyTorch + CUDA

Il comando sopra potrebbe installare una versione di `torch` **senza supporto CUDA** (versione CPU-only). Se poi lo script dà errore relativo a CUDA non disponibile, disinstalla e reinstalla torch specificando la versione CUDA corretta per la tua GPU, ad esempio (per CUDA 12.1):

```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Controlla sul sito ufficiale di PyTorch ([pytorch.org](https://pytorch.org/get-started/locally/)) quale comando corrisponde alla versione CUDA installata sul tuo PC (puoi verificarla con `nvidia-smi` da terminale/prompt dei comandi).

Per verificare che PyTorch veda la GPU, esegui in Python:

```python
import torch
print(torch.cuda.is_available())  # deve stampare True
```

Se stampa `False`, il problema è nell'installazione di CUDA/torch, non nello script.

---

## 4. Cosa modificare nello script prima di usarlo

Nel file ci sono dei placeholder tra parentesi quadre `[...]` che **devi sostituire tu** prima di eseguire:

| Placeholder | Dove si trova | Cosa metterci |
|---|---|---|
| `"[nome.audio]"` | variabile `FILES` | Il percorso (o i percorsi) del/dei file audio da trascrivere, es. `"lezione1.mp3"` oppure percorso completo `"C:/Users/tuonome/Desktop/lezione1.mp3"`. Puoi aggiungere più file separati da virgola per elaborarli in sequenza. |
| `"[nome del corso]"` | variabile `PROMPT` | Il nome del corso/materia (es. `"Farmacologia di Precisione"`), per aiutare WhisperX a riconoscere meglio termini specifici. |
| `"[concetti chiave per i vari corsi]"` | variabile `PROMPT` | Termini tecnici, nomi propri, acronimi ricorrenti nella lezione (es. nomi di farmaci, geni, professori) — aiuta la trascrizione a riconoscerli correttamente invece di "inventare" parole simili (lo trovi nei file inseriti a disposizione. |
| `"[nome del prof/prof.ssa]"` | funzione `build_pdf` | Il nome del docente, che comparirà come sottotitolo nella prima pagina del PDF. |
| `"[nome del file pdf da creare]"` | variabile `OUTPUT_PDF` | Il nome (e opzionalmente percorso) del PDF finale, es. `"Trascrizioni_Farmacologia.pdf"`. |

**Attenzione:** `PROMPT` è definito ma nello script attuale **non viene mai passato** a `model.transcribe(...)` nella funzione `trascrivi_file`. Se vuoi che il prompt influenzi davvero la trascrizione, va aggiunto esplicitamente come parametro alla chiamata (attualmente ha effetto nullo così com'è — è un punto da correggere se ti serve questa funzionalità).

---

## 5. Come funziona lo script (in breve)

1. **Fase 1 — Demucs**: se `RIDUCI_RUMORE_DEMUCS = True`, ogni file audio viene processato da Demucs per isolare la voce dal rumore di sottofondo. Il risultato viene salvato in una cartella `demucs_output/`. Se Demucs fallisce, lo script usa comunque il file audio originale senza bloccarsi.
2. **Fase 2 — WhisperX**: il modello `large-v3` viene caricato sulla GPU e trascrive ogni file audio (pulito o originale). Se `ALLINEAMENTO = True`, viene fatto anche un allineamento parola per parola (utile per timestamp precisi).
3. **Output**:
   - Un file `.txt` di backup per ogni audio (stesso nome del file audio, con suffisso `_whisperx.txt`).
   - Un unico **PDF finale** con tutte le trascrizioni, con titolo, nome del/la prof/prof.ssa e una pagina per ogni file.

---

## 6. Tempistiche

Con `large-v3` e beam search alto (`BEAM_SIZE=10`, `BEST_OF=10`), la trascrizione è **accurata ma lenta**. Per una lezione di 1-2 ore, aspettati un tempo di elaborazione totale (Demucs + trascrizione) che può andare da qualche minuto a 20-30 minuti a seconda della GPU. Se vuoi velocizzare a scapito di un po' di precisione, riduci `BEAM_SIZE` e `BEST_OF` (es. a 5) oppure usa un modello più piccolo.

---

## 7. Problemi comuni

- **Errore CUDA out of memory**: riduci `BATCH_SIZE` (es. da 8 a 4 o 2), oppure usa un modello più piccolo (`"medium"`), oppure chiudi altri programmi che usano la GPU.
- **Demucs molto lento la prima volta**: la prima esecuzione scarica i pesi del modello `htdemucs_ft` da internet, le successive saranno più veloci.
- **`ffmpeg` non trovato**: whisperx/demucs richiedono `ffmpeg` installato nel sistema (non solo il pacchetto Python). Su Windows, scaricalo da [ffmpeg.org](https://ffmpeg.org/download.html) e aggiungilo al PATH di sistema.
- **File audio non trovato**: controlla che il percorso in `FILES` sia corretto e che il file esista davvero in quel percorso.

---

## 8. Riepilogo rapido — checklist prima di lanciare

- [ ] Python 3.11.9 installato, progetto PyCharm con venv attivo
- [ ] `pip install whisperx reportlab demucs torch torchaudio soundfile` eseguito
- [ ] GPU NVIDIA con driver + CUDA funzionanti (`torch.cuda.is_available()` → `True`)
- [ ] `ffmpeg` installato a livello di sistema
- [ ] Tutti i placeholder `[...]` sostituiti con i tuoi valori reali
- [ ] File audio presenti nel percorso indicato in `FILES`
