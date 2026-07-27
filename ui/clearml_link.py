"""ui/clearml_link.py -- put a clickable link to the Streamlit page INSIDE the ClearML web UI.

WHAT THIS IS / ISN'T
    ClearML's open-source UI cannot HOST the streamlit page. the closest thing is a clickable LINK
    from ClearML to the page, so you can reach the picker from the ClearML browser without copy-paste.
    this script makes ONE small ClearML task and registers the page URL as a clickable ARTIFACT --
    the one place ClearML OSS renders a real clickable hyperlink. open the task's ARTIFACTS tab and
    click. (report_media was the wrong home: it treats a url as a media FILE and tries to load it as
    an image -> 'Unable to load image'.)

    it only creates a task + a link (metadata) -- no agent, no GCS write, no training.

RUN IT  (needs your clearml.conf; from inside final_pipeline):
    final_venv/bin/python ui/clearml_link.py --url http://localhost:8501
    #   --url   where streamlit serves the page (default http://localhost:8501)
    #   --name  the task name shown in ClearML
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C   # noqa: E402


def main():
    # clearml is imported inside main so `--help` works without a clearml.conf.
    from clearml import Task

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501", help="where streamlit serves the page")
    ap.add_argument("--name", default="Feature Selection UI  (open me)", help="task name in ClearML")
    a = ap.parse_args()

    # a plain metadata task -- no framework hooks, no output_uri, so it writes nothing to GCS.
    task = Task.init(project_name=C.CLEARML_PROJECT, task_name=a.name,
                     task_type=Task.TaskTypes.custom, auto_connect_frameworks=False)

    # THE clickable link. upload_artifact with an http(s) url REGISTERS it as a link (it uploads
    # nothing -- ClearML detects the url scheme). the ARTIFACTS tab then shows it as a clickable
    # open action. this is the one reliable clickable hyperlink on the free/OSS tier.
    task.upload_artifact(name="Open the feature-selection UI", artifact_object=a.url)

    # a short note in the INFO panel for context (plain text)
    task.set_comment(f"Feature-selection UI: {a.url}\n"
                     "runs locally (streamlit). tick features -> build -> run. the training it "
                     "triggers is recorded back here in ClearML. the link opens only while streamlit "
                     "is running on the machine you are browsing from.")

    print(f"created ClearML task '{a.name}' in project '{C.CLEARML_PROJECT}'.")
    print(f"in the ClearML web UI open this task -> ARTIFACTS tab -> "
          f"'Open the feature-selection UI' is a clickable link to {a.url}.")
    task.close()


if __name__ == "__main__":
    main()
