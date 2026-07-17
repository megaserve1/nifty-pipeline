# Multi-PC pipeline runbook — one ClearML account, data on GCP

Everything below is verified against how PC1 is already set up (which works). To add a machine,
you make it look like PC1. To run, you drive from one controller.

---

## The architecture (what talks to what)

```
        ClearML SaaS  (app.clear.ml)          GitHub  (megaserve1/nifty-pipeline)
        = orchestration + METADATA only       = the CODE (agents clone it)
                     ^                                    ^
                     |                                    |
   PC1 (controller + agent)  ----+----  PC2 (agent)  ----+----  PC3 (agent) ...
        you run `publish` here    |         |                     |
                                  +---------+---------------------+
                                            v
                              GCS bucket  (gs://<bucket>)
                              = the DATA + models  (stays on YOUR GCP)
```

Three separate channels, and this is the whole design:
- **ClearML SaaS** holds only metadata (task graph, params, metrics). No market data ever leaves.
- **GCS bucket** holds the actual dataset + models. Data stays on your GCP.
- **GitHub** holds the code. Agents `git clone` it to run a task.

**Parallelism = number of agents.** Every machine running `clearml-agent daemon --queue training`
is a worker in one shared pool. 3 machines -> the 3 models train at once.

---

## Per-machine bootstrap  (run ONCE on each NEW PC)

```bash
# 1. CODE
git clone https://github.com/megaserve1/nifty-pipeline.git
cd nifty-pipeline

# 2. PYTHON ENV
python3 -m venv final_venv
final_venv/bin/pip install -r requirements.txt
final_venv/bin/pip install clearml-agent        # the worker daemon (separate package)

# 3. CLEARML CREDENTIALS  (same account as PC1)
clearml-init
#   -> paste the SAME credentials block PC1 uses (app.clear.ml -> Settings -> Workspace).
#   THEN add the GCS project line (clearml-init drops it):
#   open ~/clearml.conf and inside sdk { ... } add:
#       google.storage { project: "mega-ml" }
#   (or: scp ~/clearml.conf from PC1 -- it is per-machine, NEVER commit it, it holds secrets)

# 4. GCS AUTH  (so the agent can pull the dataset)
gcloud auth application-default login
gcloud auth application-default set-quota-project mega-ml

# 5. GITHUB AUTH  (so the agent can clone the PRIVATE repo)
gh auth login            # or: gh auth setup-git   if gh is already logged in
```

That is the entire per-machine setup. Nothing else is machine-specific.

---

## Health check  — run on EVERY machine before the live run

This catches every failure we hit on the first run. All must pass.

```bash
cd ~/nifty-pipeline   # (or wherever the repo is on that machine)

# a) ClearML reaches the SAME account
grep api_server ~/clearml.conf                 # must be identical on all machines

# b) the GCS project line is present  (missing = "Project could not be determined")
grep -q 'project.*mega-ml' ~/clearml.conf && echo "GCS project OK" || echo "!! ADD google.storage.project: mega-ml"

# c) GCS auth is FRESH  (the token expires -- re-login the morning of the run)
final_venv/bin/python -c "import google.auth; c,_=google.auth.default(); print('gcs auth ok')" \
  || gcloud auth application-default login

# d) the agent can clone the private repo
git ls-remote https://github.com/megaserve1/nifty-pipeline.git >/dev/null && echo "git access OK" || echo "!! gh auth login"

# e) start the agent
clearml-agent daemon --queue training --create-queue
```

When step (e) prints "Listening to queues: training", that machine is a live worker.

---

## The controller  (PC1) — running the pipeline

Only ONE machine runs these. The agents (all machines) do the training.

```bash
cd ~/Desktop/Gourav/final_pipeline

# ONE TIME after any CODE change (the agents clone the base task's committed code):
git add -A && git commit -m "..." && git push
python trainer/register_base_trainer.py --force

# THE RUN:
python core/publish_version.py --version v3           # fixed hyperparams (fast, ~1-1.5h)
#   or
python core/publish_version.py --version v3 --tune    # HPO first, then train (uses ALL agents)
```

The 3 models spread across whatever agents are listening. `select_champion` waits for all three,
then crowns the lowest `test/trading_cost`.

---

## THE PRE-FLIGHT CHECKLIST for tomorrow  (print this)

The first run failed 4 times, each on one of these. Tick every box on every machine:

- [ ] **Code pushed + base tasks re-registered** (`register_base_trainer.py --force`) AFTER the
      last code change. Agents run the base task's committed code, not your working files.
- [ ] **Every machine's `clearml.conf` has** `google.storage.project: mega-ml`.
- [ ] **Every machine re-ran `gcloud auth application-default login` this morning** (tokens expire).
- [ ] **Every machine can clone the private GitHub repo** (`gh auth login`).
- [ ] **An agent is running on each machine** (`clearml-agent daemon --queue training`).
- [ ] **The machines will not sleep / auto-reboot** during the run (disable suspend; a mid-run
      reboot on the controller kills the run).
- [ ] **Config bucket is what you intend** — `config.py` `GCS_BUCKET` (currently the demo bucket;
      set a production bucket here if you want the data separated from the demo).

If all boxes are ticked, publish and walk away.

---

## Verifying it's actually parallel

In the ClearML UI -> Workers & Queues -> Workers: you should see one worker per machine, each
"running" a task during the run. If only one machine shows a task and the others are idle, that
machine's agent isn't on the `training` queue (check step e).

---

## Cost

- **Compute:** your own PCs -> free (electricity). No VM = no hourly bill.
- **GCS storage:** dataset + models, a few hundred MB -> ~Rs 5-20 / month.
- **GCS egress:** each agent pulls the dataset (~200 MB) once per run -> ~Rs 2 / pull.
- **ClearML SaaS:** free tier, metadata only -> Rs 0.

Nothing here runs 24/7. The only real "cost" trap is a forgotten cloud VM -- and you have none;
these are local machines.

---

## What this setup is NOT (be clear with the manager)

This is the **training / experiment cluster** — multiple machines training and tuning models on
versioned data. It is **not live trading.** Live inference needs three more things that do not
exist yet: the feature team's compute-functions (not parquets), a training-vs-live parity check,
and an execution/risk layer. Those are a separate build. This runbook gets you a reproducible,
distributed, honestly-tracked **model factory** — which is the right foundation to build serving
on later.
