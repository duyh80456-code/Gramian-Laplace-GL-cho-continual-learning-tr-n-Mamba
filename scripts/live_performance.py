"""Read-only metric follower for existing Mamba-CL logs; no model/data imports.

The notebook embeds this observer so its pinned training source and checkpoint
identity can remain unchanged. Validation is labeled only for the tuning stage.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

NUMBER = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf))'


def value(line, key, percent=False):
    match = re.search(r'\b' + re.escape(key) + r'\s*=\s*' + NUMBER + (r'\s*%' if percent else ''), line, re.I)
    if not match:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) else match.group(1)


def metric_event(line, stage):
    line = line.strip()
    if line.startswith('CHECKPOINT I/O WARNING:'):
        return 'WARNING', {}, line
    if line.startswith('STEP_METRICS '):
        raw = json.loads(line[len('STEP_METRICS '):])
        fields={key:raw[key] for key in ('task','epoch','optimizer_step','loss','train_acc') if key in raw}
        text=f"Task [{fields.get('task')}] Epoch [{fields.get('epoch')}] Step [{fields.get('optimizer_step')}]"
        if fields.get('loss') is not None:text+=f" loss={fields['loss']:.4f}"
        if fields.get('train_acc') is not None:text+=f" train_acc={fields['train_acc']:.2f}%"
        return 'TRAIN_STEP', fields, text
    score = 'val_acc' if stage == 'tune' else 'test_acc'
    task = re.search(r'Task \[\s*(\d+)\s*/\s*(\d+)\s*\]', line)
    epoch = re.search(r'Epoch \[\s*(\d+)\s*/\s*(\d+)\s*\]', line)
    fields = dict(task=int(task[1]), total_tasks=int(task[2])) if task else {}
    if epoch and 'loss=' in line:
        fields.update(epoch=int(epoch[1]), total_epochs=int(epoch[2]),
                      loss=value(line, 'loss'), train_acc=value(line, 'acc_top1', True))
        return 'TRAIN', fields, line.replace('acc_top1=', 'train_acc=')
    if 'Results of task [' in line:
        fields.update(task=int(re.search(r'Results of task \[(\d+)\]', line)[1]))
        fields[score] = value(line, 'class_inc_last_acc', True)
        return 'VALIDATION' if stage == 'tune' else 'TEST', fields, f"Task [{fields['task']}] {score}={fields[score]:.2f}%"
    if ':: Eval [' in line:
        fields[score + '_task'] = value(line, 'acc_class_inc', True)
        return 'EVAL_TASK', fields, f"{score}_task={fields[score + '_task']:.2f}%"
    if line.startswith('Traceback') or re.match(r'^[\w.]*(?:Error|Exception):', line):
        return 'ERROR', {}, line
    return None


class MetricFollower:
    def __init__(self, root, queues):
        self.root = Path(root)
        self.started = time.monotonic()
        self.states = {}
        for gpu, queue in enumerate(queues):
            for job in queue:
                key = job['stage'] + '/' + job['name']
                self.states[key] = dict(gpu=gpu, job=job['name'], stage=job['stage'],
                                        offset=0, pending=b'', inode=None, initialized=False,
                                        latest={}, summary_seen=False,
                                        has_history=(self.root / key / 'console.log').is_file(),
                                        summary_history=(self.root / key / 'summary.json').is_file())

    def emit(self, state, event, historical=False):
        kind, fields, text = event
        elapsed = int(time.monotonic() - self.started)
        stamp = f'{elapsed//3600:02d}:{elapsed//60%60:02d}:{elapsed%60:02d}'
        origin = 'previous log' if historical else 'live'
        print(f"[{stamp} GPU {state['gpu']} | {state['job']} | {kind} | {origin}] {text}", flush=True)
        state['latest'][kind] = dict(fields, log_line=text, from_previous_log=historical)

    def poll(self):
        changed = False
        for key, state in self.states.items():
            path = self.root / key / 'console.log'
            if path.is_file():
                stat = path.stat()
                inode = (stat.st_dev, stat.st_ino)
                first = not state['initialized']
                if state['inode'] != inode or stat.st_size < state['offset']:
                    state.update(offset=0, pending=b'', inode=inode)
                with path.open('rb') as stream:
                    if first:
                        start = max(0, stat.st_size - 128 * 1024)
                        stream.seek(start)
                        if start:
                            stream.readline()  # discard a partial leading line
                    else:
                        stream.seek(state['offset'])
                    chunk = stream.read()
                    state['offset'] = stream.tell()
                lines = (state['pending'] + chunk).split(b'\n')
                state['pending'] = lines.pop()
                events = [event for line in lines if (event := metric_event(line.decode('utf-8', errors='replace'), state['stage']))]
                if first:
                    # Show the latest stored values without replaying every old
                    # epoch on resume. New appended events are emitted in full.
                    latest = {event[0]: (i, event) for i, event in enumerate(events)}
                    events = [event for i, event in sorted(latest.values())]
                    if not events:
                        print(f"[GPU {state['gpu']} | {state['job']}] Waiting for loss/accuracy metrics.", flush=True)
                for event in events:
                    self.emit(state, event, historical=first and state['has_history'])
                    changed = True
                state['initialized'] = True
            summary = self.root / key / 'summary.json'
            if summary.is_file() and not state['summary_seen']:
                measured = json.loads(summary.read_text())
                score = 'val_acc' if state['stage'] == 'tune' else 'test_acc'
                fields = {score: measured.get('AA')}
                self.emit(state, ('COMPLETE', fields, json.dumps(fields)), historical=state['summary_history'])
                state['summary_seen'] = True
                changed = True
        if changed:
            target = self.root / 'performance_latest.json'
            temporary = target.with_suffix('.json.tmp')
            snapshot = {key: {k: state[k] for k in ('gpu', 'job', 'stage', 'latest')} for key, state in self.states.items()}
            temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
            os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Provide the unchanged launcher command after --')
    follower = MetricFollower(args.root, json.loads(Path(args.jobs).read_text()))
    print('Live metrics: loss/train_acc every5 steps and each epoch; val_acc/test_acc after each task. '
          'Clock shows observer elapsed time. Existing log values are labeled previous log.', flush=True)
    stop = threading.Event()
    def watch():
        while not stop.is_set():
            try:
                follower.poll()
            except Exception as error:
                # The observer must never stop or alter the training process.
                print('PERFORMANCE OBSERVER WARNING:', str(error), flush=True)
            stop.wait(2)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        # Keep launcher's stdout/heartbeat and environment/signal behavior.
        code = subprocess.call(command)
    finally:
        stop.set()
        thread.join(timeout=5)
        try:
            follower.poll()
        except Exception as error:
            print('PERFORMANCE OBSERVER WARNING:', str(error), flush=True)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
