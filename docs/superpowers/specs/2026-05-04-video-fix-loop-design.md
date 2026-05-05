# video_fix_loop — design

**Data:** 2026-05-04
**Status:** aprovado pelo usuário, pronto para implementação

## Objetivo

Processar ~30 mil vídeos em loop (~389GB em 70 zips) do dataset HuggingFace `AdwolfCzar/looper_v4`, normalizando para:

- **30 fps** (re-encode quando necessário)
- **Duração máxima de 5 segundos** (truncar primeiros 5s)
- **Áudio preservado** quando presente
- **Output flat** em `outputs/<mesmo_nome>.mp4` + `outputs/<mesmo_nome>.txt`

Resumível, paralelizado e streaming (sem acumular 400GB em disco — instância tem 500GB).

## Restrições do ambiente (RunPod)

- 96 vCPUs (AMD EPYC 7K62)
- 515 GB RAM
- 500 GB NVMe (10 GB/s)
- 1× RTX 4500 Ada (24GB VRAM, NVENC/NVDEC) — disponível mas não obrigatória
- Rede ~836 Mbps download / 641 Mbps upload
- Dataset HF público, ~5.5 GB/zip em média

## Escopo

### Inclui
- Download streaming dos zips (random pick, paralelo)
- Extração para scratch dir
- ffprobe + ffmpeg por vídeo
- Sidecar `.txt` copiado junto com o vídeo
- Estado persistente em SQLite (resume)
- Limpeza automática de zips/extracted após processar
- Renomeação em caso de colisão de nome (sufixo numérico)
- CLI simples para iniciar/parar/retomar

### Não inclui
- Verificação de loop (aceito skip por restrição de tempo)
- Detecção/normalização de número de ciclos
- Upload para HF (script standalone separado consome `outputs/`)
- Validação de qualidade visual
- Re-download de zips com checksum mismatch além de retry simples

## Arquitetura

### Pipeline streaming

```
HF dataset (70 zips, 389GB)
        │
        ▼
[Downloader pool: 2 workers]
   random pick zip pendente
   → work/zips/<chunk>.zip
        │
        ▼
[Extractor pool: 2 workers]
   unzip → work/extracted/<chunk>/
   registra vídeos+txt no SQLite
        │
        ▼
[Encoder pool: ~32 processes]
   ffprobe + ffmpeg
   → outputs/<name>.mp4 + outputs/<name>.txt
        │
        ▼
[Cleaner: 1 worker]
   chunk 100% processado →
   deleta work/zips/<chunk>.zip e work/extracted/<chunk>/
```

Comunicação via `multiprocessing.Queue` (downloaded → extracted → encoded → cleanup).
Limites de fila controlam back-pressure (ex: max 4 zips baixados aguardando extração).

### Componentes

| Módulo | Responsabilidade |
|--------|------------------|
| `src/db.py` | Schema SQLite + CRUD com `WAL` mode + lock seguro |
| `src/downloader.py` | Download zip via `huggingface_hub` (suporta resume parcial), valida `.sha256` quando disponível |
| `src/extractor.py` | `zipfile.extract`, descobre pares `.mp4/.txt`, registra no DB |
| `src/encoder.py` | `ffprobe` (streams + duration + fps + audio) → decide `cp` / stream-copy / re-encode |
| `src/pipeline.py` | Orquestrador: spawn workers, queues, signal handling (SIGINT/SIGTERM gracioso) |
| `src/main.py` | CLI: `run`, `status`, `reset` |

### Schema SQLite (`state.db`)

```sql
CREATE TABLE zips (
    name        TEXT PRIMARY KEY,
    size_bytes  INTEGER,
    status      TEXT NOT NULL,    -- pending|downloading|downloaded|extracting|extracted|processing|done|failed
    error       TEXT,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE TABLE videos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    zip_name        TEXT NOT NULL,
    internal_path   TEXT NOT NULL,
    output_name     TEXT,         -- nome final (com sufixo se colidiu)
    fps_in          REAL,
    duration_in     REAL,
    has_audio       INTEGER,
    action          TEXT,         -- cp|stream_copy|reencode
    status          TEXT NOT NULL,-- pending|done|failed
    error           TEXT,
    UNIQUE(zip_name, internal_path)
);

CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_zips_status ON zips(status);
```

PRAGMA: `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`.

### FFmpeg strategy

Para cada vídeo, ffprobe primeiro (rápido, ~10ms):

| Condição | Ação | Tool |
|----------|------|------|
| `fps == 30` AND `duration <= 5` | copia binário direto | `shutil.copy` |
| `fps == 30` AND `duration > 5` | stream copy com cut | `ffmpeg -ss 0 -i in -t 5 -c copy` |
| `fps != 30` | re-encode | `ffmpeg -i in -vf fps=30 -t 5 -c:v libx264 -preset veryfast -crf 20 -c:a copy` |

- **Áudio:** sempre `-c:a copy` quando re-encoda; mantém intacto no stream copy/cp.
- **Codec vídeo:** libx264 CPU. Justificativa: 96 cores → ~32 ffmpegs paralelos saturam tudo. NVENC tem 1 hardware encoder com ~3-4 streams simultâneos viáveis. Throughput total CPU > GPU para esse volume de clips curtos.
- **GPU como fallback opcional:** flag `--use-gpu` no CLI usa `h264_nvenc -preset p4 -cq 22` se preferir.

### Random pick + concorrência

Workers fazem:
```sql
BEGIN IMMEDIATE;
UPDATE zips SET status='downloading', started_at=NOW
WHERE name = (SELECT name FROM zips WHERE status='pending'
              ORDER BY RANDOM() LIMIT 1)
RETURNING name;
COMMIT;
```

WAL + `BEGIN IMMEDIATE` previne race entre workers.

### Resume

Ao iniciar:
1. Sincroniza tabela `zips` com listagem HF (insere zips novos como `pending`).
2. Workers começam a consumir.
3. Zip em status intermediário (`downloading`, `extracting`, `processing`) ao reiniciar é resetado para o estado seguro anterior — ex: `downloading` → `pending`, `extracting` → `downloaded` (se zip ainda existe), `processing` continua de onde parou (vídeos com status `done` não reprocessam).

### Colisão de nome (output)

Antes de gravar `outputs/<name>.mp4`:
1. Se não existe → grava normal.
2. Se existe → tenta `<name>__1.mp4`, `<name>__2.mp4`, ... até achar livre. Atomicamente cria flag-file `.lock` para evitar race.
3. `.txt` segue o mesmo nome final.

DB grava `output_name` final.

### Cleanup

Cleaner monitora: quando `COUNT(videos WHERE zip_name=X AND status IN ('done','failed')) == COUNT(videos WHERE zip_name=X)`:
- Marca zip como `done`.
- `rm work/zips/<chunk>.zip` e `rm -rf work/extracted/<chunk>/`.
- Libera disco para próximo zip.

### Sinal handling

- SIGINT (Ctrl+C): para de aceitar novos itens nas queues, deixa workers atuais terminarem o vídeo em andamento, persiste estado, sai limpo.
- SIGTERM: idem.
- Crash de worker: re-spawn automático até N tentativas; vídeo marcado como `failed` no N+1.

### Logging

- `logging` stdlib + arquivo `run.log`
- Linha por evento: `[time] [worker] [zip|video] [status] msg`
- Progress no stdout: contador de zips done / total + vídeos done / total + ETA

## Estrutura final do projeto

```
video_fix_loop/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── db.py
│   ├── downloader.py
│   ├── extractor.py
│   ├── encoder.py
│   └── pipeline.py
├── docs/
│   └── superpowers/specs/2026-05-04-video-fix-loop-design.md
├── outputs/             # vídeos finais (criado em runtime)
├── work/                # scratch (criado em runtime)
├── state.db             # SQLite (criado em runtime)
├── run.log
├── requirements.txt
├── run.sh
├── .gitignore
└── README.md
```

## Estimativa de tempo

- Download: 389GB / ~100 MB/s (limite de rede) = **~65 min**
- Encode pipeline: 30k vídeos / 32 paralelos / ~1s avg = **~17 min** (sobreposto ao download)
- Total end-to-end: **~1.5h** (download é o gargalo)

## Riscos

| Risco | Mitigação |
|-------|-----------|
| Colisões de nome reais | Sufixo `__N` automático (já no design) |
| Disco enche se cleaner falha | Limites de fila + watchdog que pausa downloader se `work/` > X GB |
| ffmpeg trava | Timeout por processo (60s) + retry |
| HF rate-limit | `huggingface_hub` já gerencia; backoff exponencial em 429 |
| Dataset adiciona zips durante run | Re-sync no início; novos zips entram como pending |

## Critério de conclusão

Run finaliza quando todos os zips estão em status `done` ou `failed`. Stats finais: total processado, falhas, tempo total. Disk em `work/` deve estar vazio ao final.
