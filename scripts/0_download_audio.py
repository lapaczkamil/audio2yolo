import tkinter as tk
from tkinter import ttk, messagebox
import argparse
import yt_dlp
import os
import sys
from yt_dlp.utils import sanitize_filename

DOWNLOAD_FOLDER = "1_raw_audio/targets"
AVAILABLE_CLASSES = ["bomba",
                     "karabin",
                     "kolumna",
                     "krab",
                     "pozar",
                     "woda"]

def normalize_time(t):
    parts = t.split(":")
    
    if len(parts) == 2:
        m, s = parts
        return f"00:{int(m):02}:{int(s):02}"
        
    if len(parts) == 3:
        h, m, s = parts
        return f"{int(h):02}:{int(m):02}:{int(s):02}"

    raise ValueError("Bad timestamp")

def download_clip():
    status_var.set("Pobieranie informacji o wideo...")
    root.update_idletasks()

    url = url_entry.get()
    folder = folder_entry.get()
    
    try:
        start = normalize_time(start_entry.get())
        end = normalize_time(end_entry.get())
    except ValueError:
        status_var.set("BŁĄD: Zły format czasu. Użyj formatu M:S lub H:M:S")
        root.update_idletasks()
        return

    if(folder not in AVAILABLE_CLASSES):
        status_var.set("BŁĄD: Wpisana klasa nie istnieje")
        root.update_idletasks()
        return

    if end <= start:
        status_var.set("BŁĄD: Podany start klipu powinien byc szybciej niz podany koniec klipu")
        root.update_idletasks()
        return
    
    download_path = os.path.join(DOWNLOAD_FOLDER, folder)
    os.makedirs(download_path, exist_ok=True)

    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'cookiefile': 'cookies.txt'}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'audio')
            safe_title = sanitize_filename(title)
    except Exception as e:
        status_var.set(f"BŁĄD pobierania info: {e}")
        root.update_idletasks()
        return

    counter = 0
    final_title = safe_title
    
    while os.path.exists(os.path.join(download_path, f"{final_title}.wav")):
        counter += 1
        final_title = f"{safe_title}_{counter}"

    status_var.set(f"Pobieranie i wycinanie: {final_title}...")
    root.update_idletasks()

    download_options = {
        'format': 'bestaudio/best',
        'outtmpl': f'{download_path}/{final_title}.%(ext)s',
        'cookiefile': 'cookies.txt',

        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],

        'postprocessor_args': [
            '-ss', start,
            '-to', end,
            '-ar', '44100',
            '-ac', '1'
        ]
    }

    try:
        with yt_dlp.YoutubeDL(download_options) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError:
        status_var.set("Pobieranie filmu zakończone (ostrzeżenie yt-dlp)")
        root.update_idletasks()
        return
    
    status_var.set(f"Zakończono! Zapisano jako: {final_title}.wav")
    root.update_idletasks()
    

root = tk.Tk()
root.title("Audio Dataset Downloader")
root.geometry("500x400")

tk.Label(root, text="YouTube URL").pack()
url_entry = tk.Entry(root, width=60)
url_entry.pack()

tk.Label(root, text="Class").pack()
folder_entry = ttk.Combobox(root, values=AVAILABLE_CLASSES)
folder_entry.pack()

tk.Label(root, text="Start (np. 1:15)").pack()
start_entry = tk.Entry(root)
start_entry.pack()

tk.Label(root, text="End (np. 1:30)").pack()
end_entry = tk.Entry(root)
end_entry.pack()

tk.Button(root, text="Add Clip", command=download_clip).pack(pady=5)

status_var = tk.StringVar()
status_var.set("")

status_label = tk.Label(root, textvariable=status_var)
status_label.pack(pady=5)

root.mainloop()