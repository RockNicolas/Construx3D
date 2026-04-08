from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Optional

try:
    import tkinter as tk
    from tkinter import scrolledtext
except ImportError:  # pragma: no cover - tkinter availability depends on Python install.
    tk = None
    scrolledtext = None


class EventLogWindow:
    def __init__(self) -> None:
        self._messages: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = tk is not None and scrolledtext is not None

        if self._enabled:
            self._thread = threading.Thread(target=self._run_window, name="Construx3DLogWindow", daemon=True)
            self._thread.start()
            self._ready_event.wait(timeout=2.0)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        if not self._enabled:
            print(entry)
            return
        self._messages.put(entry)

    def close(self) -> None:
        if not self._enabled:
            return
        self._stop_event.set()
        self._messages.put("__CLOSE__")
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def _run_window(self) -> None:
        root = tk.Tk()
        root.title("Construx3D - Atividades")
        root.geometry("560x360+40+40")
        root.configure(bg="#131722")

        header = tk.Label(
            root,
            text="Atividades em tempo real",
            bg="#131722",
            fg="#f3f6ff",
            font=("Segoe UI Semibold", 12),
            anchor="w",
            padx=14,
            pady=10,
        )
        header.pack(fill="x")

        output = scrolledtext.ScrolledText(
            root,
            wrap="word",
            bg="#0b0f17",
            fg="#d8e1f0",
            insertbackground="#d8e1f0",
            relief="flat",
            borderwidth=0,
            font=("Consolas", 10),
            padx=12,
            pady=12,
        )
        output.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        output.configure(state="disabled")

        def append_message(message: str) -> None:
            output.configure(state="normal")
            output.insert("end", message + "\n")
            output.see("end")
            output.configure(state="disabled")

        def drain_queue() -> None:
            while True:
                try:
                    message = self._messages.get_nowait()
                except queue.Empty:
                    break

                if message == "__CLOSE__":
                    root.destroy()
                    return

                append_message(message)

            if self._stop_event.is_set():
                root.destroy()
                return

            root.after(80, drain_queue)

        def on_close() -> None:
            self._stop_event.set()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        self._ready_event.set()
        root.after(80, drain_queue)
        root.mainloop()
