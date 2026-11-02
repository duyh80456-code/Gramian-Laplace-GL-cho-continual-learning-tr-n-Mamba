"""Step timing and non-blocking GPU utilization sampling; no model mutations."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from .io import atomic_json
from .storage import StoragePause


def disk_status(working):
    working = Path(working)
    used = 0
    for directory, dirs, files in os.walk(working, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    used += path.stat().st_size
            except FileNotFoundError:
                pass
    disk = shutil.disk_usage(working)
    return dict(working_used_bytes=used, filesystem_used_bytes=disk.used,
                free_bytes=disk.free, total_bytes=disk.total)


def check_free(working):
    reserve = int(os.environ.get('GL_MIN_FREE_BYTES', str(2 * 1024**3)))
    if shutil.disk_usage(working).free < reserve:
        raise StoragePause('Less than configured free-space reserve (default2GiB); '
                           'retain last epoch checkpoint and save this Output before resuming.')


class GPUUtilization:
    def __init__(self):
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.values = []
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()

    def poll(self):
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        while not self.stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '-i', gpu,
                    '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    value = float(result.stdout.strip().splitlines()[0])
                    with self.lock:
                        self.values.append(value)
            except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
                pass
            self.stop.wait(1)

    def drain(self):
        with self.lock:
            values, self.values = self.values, []
        return sum(values)/len(values) if values else None

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


class StepMeter:
    def __init__(self, out, arm, memory, utilization=None):
        self.out, self.arm = Path(out), arm
        self.memory = memory
        self.utilization = utilization
        self.step = 0
        self.total_images = 0
        self.total_seconds = 0.
        self.total_wait = 0.
        self.window = []

    def record(self, task, epoch, step_in_epoch, images, wait, compute, loss=None):
        self.step += 1
        self.total_images += images
        self.total_seconds += wait + compute
        self.total_wait += wait
        self.window.append((images, wait, compute))
        seconds = sum(w+c for n,w,c in self.window)
        waiting = sum(w for n,w,c in self.window)
        event = dict(task=task+1, epoch=epoch, task_complete=False,
                     optimizer_step=self.step, step_in_epoch=step_in_epoch,
                     loss=loss,
                     progress_scope='live steps; resume uses latest committed epoch checkpoint',
                     seconds_per_step=seconds/len(self.window),
                     images_per_second=sum(n for n,w,c in self.window)/seconds,
                     dataloader_wait_seconds_per_step=waiting/len(self.window),
                     dataloader_wait_fraction=waiting/seconds,
                     vram_peak_allocated_bytes=self.memory()['allocated'],
                     vram_peak_reserved_bytes=self.memory()['reserved'])
        event['vram_peak_allocated_GiB'] = event['vram_peak_allocated_bytes']/1024**3
        event['vram_peak_reserved_GiB'] = event['vram_peak_reserved_bytes']/1024**3
        if self.step % 5 == 0:
            event['gpu_utilization_percent'] = self.utilization.drain() if self.utilization else None
            util = event['gpu_utilization_percent']
            print('STEP_METRICS ' + json.dumps(dict(arm=self.arm, **event)), flush=True)
            self.window = []
        try:
            atomic_json(event, self.out/'progress.json')
        except OSError as error:
            print('PROGRESS WRITE WARNING:',str(error),flush=True)
        return event

    def summary(self):
        return dict(optimizer_steps=self.step, images=self.total_images,
                    measured_seconds=self.total_seconds,
                    seconds_per_step=self.total_seconds/self.step,
                    images_per_second=self.total_images/self.total_seconds,
                    dataloader_wait_seconds=self.total_wait,
                    dataloader_wait_fraction=self.total_wait/self.total_seconds)
