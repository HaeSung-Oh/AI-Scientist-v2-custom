import os.path as osp
import json
import argparse
import shutil
import torch
import os
import re
import sys
import yaml
import traceback
from datetime import datetime
from ai_scientist.llm import create_client

from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_bfts_config_file,
)
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.utils.token_tracker import token_tracker


CODE_MODEL_PRESETS = {
    "qwen": "ollama/qwen3:32b",
    "coder": "ollama/qwen2.5-coder:32b",
}

BACKUP_ROOT = "experiment_backups"
BACKUP_FILE_NAMES = {
    "run_console.log",
    "idea.md",
    "idea.json",
    "bfts_config.yaml",
    "unified_tree_viz.html",
    "tree_plot.html",
    "journal.json",
    "stage_progress.json",
    "best_node_id.txt",
    "token_tracker.json",
    "token_tracker_interactions.json",
    "draft_summary.json",
    "baseline_summary.json",
    "research_summary.json",
    "ablation_summary.json",
    "review_text.txt",
    "review_img_cap_ref.json",
}
BACKUP_PREFIXES = ("best_solution_",)
BACKUP_SUFFIXES = (".pdf",)


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def should_backup_file(filename):
    return (
        filename in BACKUP_FILE_NAMES
        or filename.startswith(BACKUP_PREFIXES)
        or filename.endswith(BACKUP_SUFFIXES)
    )


def backup_experiment(idea_dir, reason="manual"):
    if not idea_dir or not osp.isdir(idea_dir):
        return

    backup_dir = osp.join(BACKUP_ROOT, osp.basename(idea_dir))
    copied = 0
    os.makedirs(backup_dir, exist_ok=True)

    for root, _, files in os.walk(idea_dir):
        for filename in files:
            if not should_backup_file(filename):
                continue
            src = osp.join(root, filename)
            rel_path = osp.relpath(src, idea_dir)
            dst = osp.join(backup_dir, rel_path)
            os.makedirs(osp.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
                copied += 1
            except OSError as e:
                print(f"Warning: failed to back up {src}: {e}")

    manifest_path = osp.join(backup_dir, "backup_manifest.json")
    manifest = {
        "source": osp.abspath(idea_dir),
        "backup": osp.abspath(backup_dir),
        "reason": reason,
        "copied_files": copied,
        "timestamp": datetime.now().isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Backed up {copied} files to {backup_dir} ({reason})")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/i_cant_believe_its_not_better.json",
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as ideas file but .py extension",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="ollama/qwen3:32b",
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="ollama/qwen3:32b",
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="ollama/qwen3:32b",
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="ollama/qwen3:32b",
        help="Smaller model to use for writeup",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="ollama/qwen3:32b",
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--code",
        type=str,
        default="qwen",
        choices=sorted(CODE_MODEL_PRESETS),
        help=(
            "Code-generation model preset. "
            "'qwen' uses ollama/qwen3:32b; 'coder' uses ollama/qwen2.5-coder:32b."
        ),
    )
    parser.add_argument(
        "--code-model",
        type=str,
        default=None,
        help="Explicit code-generation model override, e.g. ollama/codestral:22b.",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def find_pdf_path_for_review(idea_dir):
    pdf_path = None
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        # First check if there's a final version
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            # Use the final version if available
            pdf_path = osp.join(idea_dir, final_pdfs[0])
        else:
            # Try to find numbered reflections
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                # Get the file with the highest reflection number
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                pdf_path = osp.join(idea_dir, highest_reflection[1])
            else:
                # Fall back to the first reflection PDF if no numbers found
                pdf_path = osp.join(idea_dir, reflection_pdfs[0])
    return pdf_path


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def tee_stdout_stderr_to_file(log_file_path):
    log = open(log_file_path, "a", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    return log


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")

    idea = ideas[args.idea_idx]

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)
    console_log_path = osp.join(idea_dir, "run_console.log")
    console_log = tee_stdout_stderr_to_file(console_log_path)
    print(f"Console output is being saved to {console_log_path}")

    original_excepthook = sys.excepthook

    def backup_then_report_exception(exc_type, exc_value, exc_traceback):
        print("Unhandled exception occurred. Creating experiment backup before exit.")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        backup_experiment(idea_dir, reason="unhandled_exception")
        original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = backup_then_report_exception

    # Convert idea json to markdown file
    idea_path_md = osp.join(idea_dir, "idea.md")

    # If load_code is True, get the Python file with same name as JSON
    code = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
        else:
            print(f"Warning: Code file {code_path} not found")
    else:
        code_path = None

    idea_to_markdown(ideas[args.idea_idx], idea_path_md, code_path)

    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            print(f"Warning: Dataset reference file {dataset_ref_path} not found")
            dataset_ref_code = None

    if dataset_ref_code is not None and code is not None:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code is not None and code is None:
        added_code = dataset_ref_code
    elif dataset_ref_code is None and code is not None:
        added_code = code
    else:
        added_code = None

    print(added_code)

    # Add code to idea json if it was loaded
    if added_code is not None:
        ideas[args.idea_idx]["Code"] = added_code

    # Store raw idea json
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    config_path = "bfts_config.yaml"
    idea_config_path = edit_bfts_config_file(
        config_path,
        idea_dir,
        idea_path_json,
    )
    code_model = args.code_model or CODE_MODEL_PRESETS[args.code]
    with open(idea_config_path, "r") as f:
        run_config = yaml.load(f, Loader=yaml.FullLoader)
    run_config["agent"]["code"]["model"] = code_model
    with open(idea_config_path, "w") as f:
        yaml.dump(run_config, f)
    print(f"Using code model: {code_model}")
    backup_experiment(idea_dir, reason="initialized")

    perform_experiments_bfts(idea_config_path)
    backup_experiment(idea_dir, reason="after_bfts")
    experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
    if os.path.exists(experiment_results_dir):
        shutil.copytree(
            experiment_results_dir,
            osp.join(idea_dir, "experiment_results"),
            dirs_exist_ok=True,
        )

    try:
        aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)
    except Exception:
        print("Plot aggregation failed. Continuing with saved experiment summaries.")
        print(traceback.format_exc())
    backup_experiment(idea_dir, reason="after_plot_aggregation")

    copied_experiment_results_dir = osp.join(idea_dir, "experiment_results")
    if osp.exists(copied_experiment_results_dir):
        shutil.rmtree(copied_experiment_results_dir)

    save_token_tracker(idea_dir)
    backup_experiment(idea_dir, reason="after_token_tracker")

    if not args.skip_writeup:
        writeup_success = False
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=8,
                    citations_text=citations_text,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    save_token_tracker(idea_dir)
    backup_experiment(idea_dir, reason="after_writeup")

    if not args.skip_review and not args.skip_writeup:
        # Perform paper review if the paper exists
        pdf_path = find_pdf_path_for_review(idea_dir)
        if pdf_path and os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")
            backup_experiment(idea_dir, reason="after_review")
        else:
            print("No paper PDF found for review. Skipping review.")

    print("Start cleaning up processes")
    # Kill all mp and torch processes associated with this experiment
    import psutil
    import signal

    # Get the current process and all its children
    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    # First try graceful termination
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Wait briefly for processes to terminate
    gone, alive = psutil.wait_procs(children, timeout=3)

    # If any processes remain, force kill them
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Additional cleanup: find any orphaned processes containing specific keywords
    keywords = ["python", "torch", "mp", "bfts", "experiment"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # Check both process name and command line arguments
            cmdline = " ".join(proc.cmdline()).lower()
            if any(keyword in cmdline for keyword in keywords):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
                if proc.is_running():
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue

    # Finally, terminate the current process
    # current_process.send_signal(signal.SIGTERM)
    # try:
    #     current_process.wait(timeout=3)
    # except psutil.TimeoutExpired:
    #     current_process.kill()

    # exit the program
    backup_experiment(idea_dir, reason="final")
    console_log.close()
    sys.exit(0)
