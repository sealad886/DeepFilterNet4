import sys
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class TrainingDashboard:
    def __init__(self, disable: bool = False):
        self.console = Console(stderr=True)
        self.disable = disable

        if self.disable:
            self.live = None
            return

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[metrics]}"),
            console=self.console,
        )

        self.setup_text = Text("")
        self.metrics_text = Text("")

        self.group = Group(
            Panel(self.setup_text, title="Setup", border_style="blue"),
            Panel(self.metrics_text, title="Metrics", border_style="green"),
            self.progress,
        )

        self.live = Live(self.group, console=self.console, refresh_per_second=4)

        self.train_task: Optional[TaskID] = None
        self.valid_task: Optional[TaskID] = None
        self.epoch_task: Optional[TaskID] = None

    def start(self):
        if self.live:
            self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def write(self, message: str = ""):
        if self.disable:
            print(message, file=sys.stderr)
        elif self.live:
            self.console.print(message)

    def update_setup(self, info: str):
        self.setup_text.plain = info

    def update_metrics(self, metrics: str):
        self.metrics_text.plain = metrics

    def start_epoch(self, epoch: int, total_epochs: int):
        if self.disable:
            return
        if self.epoch_task is None:
            self.epoch_task = self.progress.add_task("[bold blue]Epochs", total=total_epochs, metrics="")
        self.progress.update(self.epoch_task, completed=epoch)

    def start_train(self, total: int, description: str = "[bold green]Training"):
        if self.disable:
            return
        if self.train_task is None:
            self.train_task = self.progress.add_task(description, total=total, metrics="")
        else:
            self.progress.reset(self.train_task, total=total, description=description, metrics="")
            self.progress.start_task(self.train_task)

    def update_train(self, advance: int = 1, metrics: str = ""):
        if self.disable or self.train_task is None:
            return
        self.progress.update(self.train_task, advance=advance, metrics=metrics)

    def start_valid(self, total: int, description: str = "[bold yellow]Validation"):
        if self.disable:
            return
        if self.valid_task is None:
            self.valid_task = self.progress.add_task(description, total=total, metrics="")
        else:
            self.progress.reset(self.valid_task, total=total, description=description, metrics="")
            self.progress.start_task(self.valid_task)

    def update_valid(self, advance: int = 1, metrics: str = ""):
        if self.disable or self.valid_task is None:
            return
        self.progress.update(self.valid_task, advance=advance, metrics=metrics)
